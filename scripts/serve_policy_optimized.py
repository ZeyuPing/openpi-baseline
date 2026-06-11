#!/usr/bin/env python3
"""Optimized policy server with temporal ensembling and tunable denoising.

Three inference-time optimizations over the vanilla serve_policy.py:

  1. **Denoising steps**  (--num-denoise-steps, default 20)
     Flow matching quality scales with more Euler steps.  Default 10 → 20
     gives notably better action predictions at modest latency cost.

  2. **Short closed-loop horizon**  (--execute-horizon, default 20)
     The model always outputs a full 50-step chunk, but only the first K
     steps are returned to the client.  Clients that consume
     ``min(client_horizon, len(actions))`` will therefore re-query after K
     steps even if their own default horizon is 50.  The remaining model
     steps overlap with the next query and are used for temporal ensembling.

  3. **Noise-space ensembling**  (--noise-ensembling, default True)
     The overlapping part of the flow-matching noise sequence is shifted
     forward across queries.  Action-space blending is disabled by default
     (--blend-alpha 0.0) so each returned action comes from the current
     observation's model output.

Usage:
  uv run scripts/serve_policy_optimized.py \\
      --config pi05_multitask-positive \\
      --dir checkpoints/pi05_multitask-positive/<exp>/<step> \\
      --port 8000 \\
      --num-denoise-steps 20 \\
      --execute-horizon 20 \\
      --return-horizon 20 \\
      --temporal-ensembling

  # Disable temporal ensembling (pure denoising upgrade only):
  uv run scripts/serve_policy_optimized.py \\
      --config pi05_multitask-generalist \\
      --dir checkpoints/pi05_multitask-generalist/<exp>/<step> \\
      --no-temporal-ensembling
"""

from __future__ import annotations

import dataclasses
import logging
import socket
import time
from typing import Any

import numpy as np
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

logger = logging.getLogger("serve_optimized")


# ---------------------------------------------------------------------------
# Temporal Ensembling wrapper
# ---------------------------------------------------------------------------

class TemporalEnsemblingPolicy:
    """Wraps a base policy with short-horizon serving and optional ensembling.

    On each ``infer()`` call the wrapper:
      1. Calls the underlying policy to produce a full action chunk (50, D).
      2. Optionally reuses shifted flow-matching noise for the overlapping
         region across queries.
      3. If ``blend_alpha > 0`` and a cached remainder from the previous call
         exists, blends the overlapping region using linearly decaying weights.
         - Joint dimensions are blended.
         - Gripper dimensions (6, 13) are NOT blended to preserve sharp
           open/close transitions.
      4. Returns only the first ``return_horizon`` steps. This forces clients
         that consume ``min(client_horizon, len(actions))`` to re-query after
         the returned short chunk.

    Timeline illustration (execute_horizon K = 20, action_horizon H = 50):

        Query 1:  [████████ execute ████████][───────── cache ─────────]
                  step 0                   19 20                      49

        Query 2:                             [■■■■■■ new chunk (50) ■■■■■■]
                  overlap:                   [blend 30 steps]
                  cached_old[0:30]  ←blend→  new_chunk[0:30]
                                             [████ execute ████][── cache ──]
    """

    # Gripper dims in the 14-dim YAM action layout.
    GRIPPER_DIMS = frozenset({6, 13})

    def __init__(
        self,
        base_policy: Any,
        execute_horizon: int = 20,
        return_horizon: int | None = None,
        blend_alpha: float = 0.0,
        noise_ensembling: bool = True,
    ) -> None:
        """
        Args:
            base_policy:     The underlying openpi ``Policy`` object.
            execute_horizon: How many steps the client will execute before
                             re-querying. We enforce this by returning at most
                             this many actions to clients that respect the
                             returned chunk length.
            return_horizon:  How many actions to return over websocket. If not
                             set, defaults to ``execute_horizon``.
            blend_alpha:     Maximum weight given to the *cached* (old) actions
                             in the overlap region.  The weight decays linearly
                             to 0 at the end of the overlap so that far-future
                             steps are dominated by the fresh prediction. Set
                             to 0.0 to disable action-space blending and keep
                             the current model output untouched.
            noise_ensembling: Enable consistent noise sampling across queries.
        """
        self._base = base_policy
        self._execute_horizon = execute_horizon
        self._return_horizon = return_horizon if return_horizon is not None else execute_horizon
        self._blend_alpha = blend_alpha
        self._noise_ensembling = noise_ensembling

        if self._execute_horizon <= 0:
            raise ValueError(f"execute_horizon must be positive, got {self._execute_horizon}")
        if self._return_horizon <= 0:
            raise ValueError(f"return_horizon must be positive, got {self._return_horizon}")
        if not 0.0 <= self._blend_alpha <= 1.0:
            raise ValueError(f"blend_alpha must be in [0, 1], got {self._blend_alpha}")
        if self._return_horizon != self._execute_horizon:
            raise ValueError(
                "return_horizon must match execute_horizon so the temporal cache "
                "aligns with the actions the client actually executes "
                f"(got return_horizon={self._return_horizon}, execute_horizon={self._execute_horizon})"
            )

        # Cache of remaining unreturned/unexecuted actions from the previous
        # model chunk, shape (overlap_len, action_dim) or None.
        self._cached_actions: np.ndarray | None = None
        self._prev_noise: np.ndarray | None = None
        self._query_count: int = 0
        self._last_query_time: float = 0.0

        # Retrieve action horizon and action dimension from the base policy/model config.
        model = getattr(base_policy, "_model", None)
        self._action_horizon = 50
        self._action_dim = 14
        if model is not None:
            self._action_horizon = getattr(model, "action_horizon", self._action_horizon)
            self._action_dim = getattr(model, "action_dim", self._action_dim)
            if hasattr(model, "config"):
                self._action_horizon = getattr(model.config, "action_horizon", self._action_horizon)
                self._action_dim = getattr(model.config, "action_dim", self._action_dim)
        if self._execute_horizon > self._action_horizon:
            raise ValueError(
                f"execute_horizon ({self._execute_horizon}) cannot exceed model action_horizon "
                f"({self._action_horizon})"
            )
        if self._return_horizon > self._action_horizon:
            raise ValueError(
                f"return_horizon ({self._return_horizon}) cannot exceed model action_horizon ({self._action_horizon})"
            )

    # -- public API expected by WebsocketPolicyServer -----------------------

    @property
    def metadata(self) -> dict[str, Any]:
        if hasattr(self._base, "metadata"):
            return self._base.metadata
        return {}

    def infer(self, obs: dict) -> dict:
        now = time.monotonic()

        # Auto-reset if idle for >30 s (likely a new evaluation episode).
        if self._last_query_time > 0 and (now - self._last_query_time) > 30.0:
            logger.info("Idle >30 s — resetting temporal ensembling cache")
            self.reset()
        self._last_query_time = now

        # 1. Prepare consistent noise sequence (Noise-Space Temporal Ensembling)
        noise_arg = None
        if self._noise_ensembling:
            if self._prev_noise is None:
                current_noise = self._sample_noise(self._action_horizon)
            else:
                current_noise = np.empty((self._action_horizon, self._action_dim), dtype=np.float32)
                overlap_len = self._action_horizon - self._execute_horizon
                if overlap_len > 0:
                    current_noise[0:overlap_len] = self._prev_noise[self._execute_horizon:self._action_horizon]
                    current_noise[overlap_len:] = self._sample_noise(self._execute_horizon)
                else:
                    current_noise = self._sample_noise(self._action_horizon)
            self._prev_noise = current_noise
            noise_arg = current_noise

        # 2. Forward pass through the underlying policy, injecting consistent noise.
        result = self._base.infer(obs, noise=noise_arg)
        new_actions = np.asarray(result["actions"], dtype=np.float64)
        action_dim = new_actions.shape[-1]

        # 3. Optionally blend with cached remainder. The default alpha is 0.0
        # so closed-loop corrections from the fresh observation are untouched.
        if self._blend_alpha > 0.0 and self._cached_actions is not None:
            overlap_len = min(len(self._cached_actions), len(new_actions))
            if overlap_len > 0:
                self._blend_in_place(new_actions, self._cached_actions, overlap_len, action_dim)
                if self._query_count <= 3 or self._query_count % 20 == 0:
                    logger.info(
                        "query %d: blended %d overlap steps (alpha=%.2f, noise_ensembling=%s)",
                        self._query_count, overlap_len, self._blend_alpha, self._noise_ensembling
                    )

        # 4. Cache overlap actions only when action-space blending is enabled.
        K = self._execute_horizon
        if self._blend_alpha > 0.0 and K < len(new_actions):
            self._cached_actions = new_actions[K:].copy()
        else:
            self._cached_actions = None

        self._query_count += 1
        result["actions"] = new_actions[: self._return_horizon].astype(np.float32, copy=False)
        return result

    def reset(self) -> None:
        """Clear the cache (e.g. between episodes)."""
        self._cached_actions = None
        self._prev_noise = None
        self._query_count = 0
        logger.info("Temporal ensembling cache reset")

    # -- internal -----------------------------------------------------------

    def _blend_in_place(
        self,
        new: np.ndarray,
        old: np.ndarray,
        overlap_len: int,
        action_dim: int,
    ) -> None:
        """Blend ``new[0:overlap_len]`` with ``old[0:overlap_len]`` in-place.

        - Joint dims: weighted average with linearly decaying alpha.
        - Gripper dims (6, 13): keep the new prediction untouched.
        """
        for i in range(overlap_len):
            # Linear decay: full alpha at step 0, zero at the end of overlap.
            decay = 1.0 - i / overlap_len
            alpha = self._blend_alpha * decay  # weight for OLD cached value

            for d in range(min(action_dim, 14)):
                if d in self.GRIPPER_DIMS:
                    continue  # keep new prediction for grippers
                new[i, d] = (1.0 - alpha) * new[i, d] + alpha * old[i, d]

    def _sample_noise(self, length: int) -> np.ndarray:
        return np.random.normal(size=(length, self._action_dim)).astype(np.float32)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def _build_metadata(base_metadata: dict[str, Any] | None, return_horizon: int) -> dict[str, Any]:
    """Advertise the chunk length and YAM I/O contract expected by policy_deployment."""
    metadata = dict(base_metadata or {})
    metadata.update(
        {
            "protocol_version": metadata.get("protocol_version", "1.0"),
            "policy_name": metadata.get("policy_name", "openpi-pi05-yam"),
            "control_mode": metadata.get("control_mode", "joints"),
            "action_horizon": return_horizon,
            "action_dim": 14,
            "state_dim": 14,
            "image_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            "image_shape": metadata.get("image_shape", [3, 224, 224]),
            "expects_prompt": True,
        }
    )
    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Args:
    """Optimized policy server arguments."""

    # ── Model ──────────────────────────────────────────────────────────────
    # Training config name (e.g. "pi05_multitask-positive").
    config: str

    # Checkpoint directory (e.g. "checkpoints/pi05_multitask-positive/pi05_multitask-positive/80000").
    dir: str

    # Port to serve the policy on.
    port: int = 8000

    # Fallback prompt when the client doesn't send one.
    default_prompt: str | None = None

    # ── Denoising ──────────────────────────────────────────────────────────
    # Number of Euler steps in the flow-matching denoising loop.
    # Default in openpi is 10; 20 gives better quality at ~2× latency.
    num_denoise_steps: int = 20

    # ── Temporal ensembling ────────────────────────────────────────────────
    # Enable / disable temporal ensembling.
    temporal_ensembling: bool = True

    # Enable / disable noise-space ensembling (consistent noise sampling).
    # Reuses and shifts the noise vector for the overlapping steps of flow-matching denoising.
    # Highly recommended for physics/mode consistency.
    noise_ensembling: bool = True

    # How many steps the client will execute before re-querying.
    # For third-party clients like policy_deployment/check_in_sim.py, this is
    # enforced by returning only this many actions.
    execute_horizon: int = 20

    # How many action rows to return to the websocket client. Keep this equal
    # to execute_horizon so the cached overlap matches the executed prefix.
    return_horizon: int = 20

    # Weight for cached (old) actions at the start of the overlap region.
    # Default 0 disables action-space blending to preserve closed-loop correction.
    blend_alpha: float = 0.0


def main(args: Args) -> None:
    train_config = _config.get_config(args.config)

    # ── Create policy with custom denoising steps ──────────────────────────
    logger.info("Creating policy  config=%s  dir=%s", args.config, args.dir)
    logger.info("Denoising steps: %d  (default is 10)", args.num_denoise_steps)

    policy = _policy_config.create_trained_policy(
        train_config,
        args.dir,
        default_prompt=args.default_prompt,
        sample_kwargs={"num_steps": args.num_denoise_steps},
    )
    advertised_horizon = args.return_horizon if args.temporal_ensembling else train_config.model.action_horizon
    policy_metadata = _build_metadata(policy.metadata, advertised_horizon)

    # ── Wrap with temporal ensembling ──────────────────────────────────────
    if args.temporal_ensembling:
        policy = TemporalEnsemblingPolicy(
            policy,
            execute_horizon=args.execute_horizon,
            return_horizon=args.return_horizon,
            blend_alpha=args.blend_alpha,
            noise_ensembling=args.noise_ensembling,
        )
        logger.info(
            "Temporal ensembling ON: execute_horizon=%d return_horizon=%d blend_alpha=%.2f noise_ensembling=%s",
            args.execute_horizon, args.return_horizon, args.blend_alpha, args.noise_ensembling,
        )
    else:
        logger.info("Temporal ensembling OFF")

    # ── Launch server ──────────────────────────────────────────────────────
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info("Starting server  host=%s  ip=%s  port=%d", hostname, local_ip, args.port)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    main(tyro.cli(Args))
