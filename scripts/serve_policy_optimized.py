#!/usr/bin/env python3
"""Optimized policy server with temporal ensembling and tunable denoising.

Three inference-time optimizations over the vanilla serve_policy.py:

  1. **Denoising steps**  (--num-denoise-steps, default 20)
     Flow matching quality scales with more Euler steps.  Default 10 → 20
     gives notably better action predictions at modest latency cost.

  2. **Execute horizon**  (--execute-horizon, default 20)
     The model always outputs a full 50-step chunk, but only the first K
     steps are intended to be executed by the client.  The remaining steps
     overlap with the next query and are used for temporal ensembling.
     ⚠ This value MUST match the client's --action-horizon flag.

  3. **Temporal ensembling**  (--temporal-ensembling, default True)
     When the server receives the next query, the newly predicted chunk is
     blended with the *cached remainder* of the previous chunk in the
     overlapping region.  Joint dims are blended with exponentially decaying
     weights; gripper dims (6, 13) are NOT blended to preserve sharp
     open/close transitions.

Usage:
  uv run scripts/serve_policy_optimized.py \\
      --config pi05_multitask-positive \\
      --dir checkpoints/pi05_multitask-positive/<exp>/<step> \\
      --port 8000 \\
      --num-denoise-steps 20 \\
      --execute-horizon 20 \\
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
    """Wraps a base policy with temporal ensembling of action chunks.

    On each ``infer()`` call the wrapper:
      1. Calls the underlying policy to produce a full action chunk (50, D).
      2. If a cached remainder from the *previous* call exists, blends the
         overlapping region using exponentially decaying weights.
         - Joint dimensions are blended.
         - Gripper dimensions (6, 13) are NOT blended to preserve sharp
           open/close transitions.
      3. Caches steps [execute_horizon:] for the *next* call.
      4. Returns the full (blended) chunk — the client controls how many
         steps to actually execute via its own ``--action-horizon``.

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
        blend_alpha: float = 0.3,
    ) -> None:
        """
        Args:
            base_policy:     The underlying openpi ``Policy`` object.
            execute_horizon: How many steps the client will execute before
                             re-querying.  Must match the client's
                             ``--action-horizon``.
            blend_alpha:     Maximum weight given to the *cached* (old) actions
                             in the overlap region.  The weight decays linearly
                             to 0 at the end of the overlap so that far-future
                             steps are dominated by the fresh prediction.
                             Recommended range: 0.2 – 0.5.
        """
        self._base = base_policy
        self._execute_horizon = execute_horizon
        self._blend_alpha = blend_alpha

        # Cache of remaining (unexecuted) actions from the previous chunk,
        # shape (overlap_len, action_dim) or None.
        self._cached_actions: np.ndarray | None = None
        self._query_count: int = 0
        self._last_query_time: float = 0.0

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

        # 1. Forward pass through the underlying policy.
        result = self._base.infer(obs)
        new_actions = np.asarray(result["actions"], dtype=np.float64)
        action_dim = new_actions.shape[-1]

        # 2. Blend with cached remainder (if any).
        if self._cached_actions is not None:
            overlap_len = min(len(self._cached_actions), len(new_actions))
            if overlap_len > 0:
                self._blend_in_place(new_actions, self._cached_actions, overlap_len, action_dim)
                if self._query_count <= 3 or self._query_count % 20 == 0:
                    logger.info(
                        "query %d: blended %d overlap steps (alpha=%.2f)",
                        self._query_count, overlap_len, self._blend_alpha,
                    )

        # 3. Cache the steps that the client will NOT execute.
        K = self._execute_horizon
        if K < len(new_actions):
            self._cached_actions = new_actions[K:].copy()
        else:
            self._cached_actions = None

        self._query_count += 1
        result["actions"] = new_actions
        return result

    def reset(self) -> None:
        """Clear the cache (e.g. between episodes)."""
        self._cached_actions = None
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

    # How many steps the client will execute before re-querying.
    # ⚠ Must match the client's --action-horizon.
    execute_horizon: int = 20

    # Weight for cached (old) actions at the start of the overlap region.
    # Decays linearly to 0 at the end of the overlap.
    # Higher → smoother transitions but slower reaction; lower → more reactive.
    blend_alpha: float = 0.3


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
    policy_metadata = policy.metadata

    # ── Wrap with temporal ensembling ──────────────────────────────────────
    if args.temporal_ensembling:
        policy = TemporalEnsemblingPolicy(
            policy,
            execute_horizon=args.execute_horizon,
            blend_alpha=args.blend_alpha,
        )
        logger.info(
            "Temporal ensembling ON:  execute_horizon=%d  blend_alpha=%.2f",
            args.execute_horizon, args.blend_alpha,
        )
        logger.info(
            "⚠  Client must use --action-horizon %d to match execute_horizon",
            args.execute_horizon,
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
