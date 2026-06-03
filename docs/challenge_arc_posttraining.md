# Challenge Takeover-Aware ARC Post-Training

> Status: **method design for Goal 1**. This document is written for coding agents who will
> produce the implementation plan next. It should be treated as the source of truth for the
> next method family, not as a description of the current branch. The official `main` baseline is
> expert-only SFT with the three `pi05_*` configs. The current development branch adds a static
> weighted SFT path (`pi05w_*`) with hard-coded source/mode weights. This proposal builds on that
> weighted path, but the method itself is additive and must remain comparable against `main`.

## 0. Corrected Dataset Assumptions

Phase 1 is offline only. Do not assume access to online rollouts, evaluator progress labels, or
hand-authored subtask divisions.

The Hugging Face dataset exposes these reliable signals:

- `expert-data`: high-quality human teleoperation demonstrations.
- `failure-data`: baseline-policy rollouts that failed. These are useful as negative/value data,
  but should not be naively imitated.
- `success-and-hil-data`: baseline-policy rollouts that succeeded, including autonomous successes
  and human-in-the-loop interventions.
- `observation.commander_state`: per-frame mode. Observed modes include `inference`, `teleop`,
  `pre_teleop`, `restore`, and rare `align`.
- `observation.state`, `action`, three camera streams, `timestamp`, `frame_index`,
  `episode_index`, and `task_index`.

Important non-signals:

- The `subtask` column is present in the schema but is filled with `TODO` in the dataset we
  inspected. Treat it as unavailable.
- The challenge's graduated progress Score is computed by the evaluator, not stored in the
  training data.
- Except for an extra `reward` field in `insert-mouse-battery/failure-data`, there is no general
  reward column. The method must not depend on that exception.

Therefore the core learning signal must come from **source provenance, terminal success/failure,
commander-state structure, and takeover boundaries**.

## 1. Method Summary

Use **Takeover-Aware Monte-Carlo Advantage-Weighted Regression** as the primary Phase 1 method:

1. Merge the task's `expert-data`, `success-and-hil-data`, and `failure-data` into a single
   LeRobot root, preserving provenance in `meta/sources.jsonl`.
2. Build valid actor chunks exactly as the current weighted path does: reject chunks that cross
   commander-state boundaries, reject drop modes, reject action discontinuities, and reject short
   tails.
3. Synthesize weak offline rewards from only reliable labels.
4. Train a value model from Monte-Carlo returns. Keep takeover risk as deterministic metadata,
   not as a learned prediction target.
5. Validate the value model offline.
6. Emit clipped AWR sample weights into the same parquet index schema used by
   `scripts/train_weighted.py`.
7. Train `pi05awr_*` policies with the existing weighted flow-matching loss.

Only after Stage A validates should we implement **Stage B: advantage-conditioned training + CFG**
(`pi05ac_*`). Stage B is higher ceiling but more invasive, so it is a gated extension rather than
the first submission bet.

In short:

```text
main baseline:          expert-only SFT
current branch:         static source/mode weighted SFT
primary proposal:       takeover-aware MC value -> clipped AWR weights
gated extension:        advantage-conditioned policy -> positive CFG inference
```

## 2. Why AWR First

pi0.5 is a flow-matching VLA, so action log-probabilities are not readily available. AWR only
reweights the existing flow-matching loss and is therefore log-prob-free:

```text
L = sum_i w_i * mean_horizon(flow_loss_i) / max(sum_i w_i, 1)
w_i = base_weight_i * clip(exp(A_i / beta), w_min, w_max)
```

This is the safest next step because:

- It reuses the existing weighted-loader/training surface.
- If the value model is uninformative, robustly standardized advantages collapse near zero and the
  method falls back toward static weighted SFT.
- It keeps failure trajectories out of positive actor imitation while still using them to train the
  value model.
- It is easy to ablate per task and per source.

Avoid offline TD/Q methods for Phase 1. The dataset is small and off-policy; without online
correction, Q bootstrapping is a high-risk failure mode.

## 3. Reward Synthesis From Available Signals

Use a simple Monte-Carlo reward definition that is task-general and does not assume subtasks.

Per episode, determine:

- `source_name` from the source leaf or merged-root provenance.
- `episode_success = True` for `expert-data` and `success-and-hil-data`.
- `episode_success = False` for `failure-data`.
- commander-state segments from `observation.commander_state`.
- takeover boundaries inside `success-and-hil-data`: an episode is HIL if it contains both
  `inference` and `teleop`.

Base reward:

```text
r_t = -1 / T_success_scale           for ordinary non-terminal kept frames
r_T = 0                              for successful terminal frame
r_T = -C_fail                        for failed terminal frame
```

Recommended implementation detail: normalize returns per task to a stable range, for example
`[-1, 0]`, so tasks with different episode lengths are comparable.

Takeover shaping:

```text
teleop frames in successful HIL episode:
  add small positive correction bonus, or force their actor advantage/label positive later

pre_teleop frames:
  assign negative risk bonus and exclude from actor imitation

inference frames immediately before teleop:
  assign negative risk bonus and downweight/exclude from actor imitation

restore / align:
  exclude from actor imitation and do not treat as task progress
```

Keep this shaping conservative. It should identify risk/correction structure, not invent subtask
labels.

## 4. Takeover Processing Contract

This section is non-negotiable for implementation. The `inference -> teleop` boundary is a real
data discontinuity: the dataset README warns that the last autonomous frame and first teleop frame
can differ by up to about `0.2` rad per joint due to GELLO/YAM alignment at takeover.

For each episode:

1. Segment contiguous runs of `observation.commander_state`.
2. Mark drop modes: `pre_teleop`, `restore`, `align`.
3. Mark HIL takeover if an `inference` segment is followed by `pre_teleop` or `teleop`.
4. Define a configurable pre-takeover risk window before each takeover, e.g. `risk_window_frames`
   in `[30, 120]` at 60 Hz.
5. Define valid actor chunks only if the whole action horizon:
   - stays within one commander-state segment,
   - is not in a drop mode,
   - does not include a large action/state jump,
   - has enough future frames for the configured horizon.

Actor semantics by region:

| Region | Actor imitation | Value training / deterministic metadata | Stage B label |
| --- | --- | --- | --- |
| `expert-data` | positive, base weight `1.0` | success return | positive |
| `success-and-hil-data`, pure `inference` success | positive, base weight `0.7` | success return | by advantage |
| `success-and-hil-data`, `teleop` | positive, base weight `2.0` | correction/success return | forced positive |
| `success-and-hil-data`, `pre_teleop` | exclude | risky pre-takeover | negative/drop |
| inference risk window before takeover | exclude, base actor weight `0.0` | risky pre-takeover | negative |
| `restore` / `align` | exclude | reset/drop region | drop |
| `failure-data` ordinary frames | default zero actor weight for Stage A | failed return | negative for Stage B |

Important: do not create action chunks that straddle `inference -> teleop`, `teleop -> inference`,
or any drop-mode boundary. This applies to both value-label generation and actor training indexes.

For value training, keep pre-takeover and failure frames because they teach the critic about risk.
For actor training, default to excluding them from positive imitation.

## 5. Value Model

Train a frozen-feature value model, not a full end-to-end actor-critic.

Inputs should include:

- token-level image/language features from the existing pi0.5/PaliGemma prefix path,
- the task prompt or task id,
- `observation.state` explicitly as numeric proprioception,
- optional commander-state embeddings for value training only.

Outputs:

- `V(s)`: normalized Monte-Carlo return / negative steps-to-success.

For the Stage A implementation, prefer the IG-RFT-style frozen-feature critic pattern: cache
pi0.5 prefix token activations and masks, aggregate them with a small learned-query cross-attention
module, then fuse the visual-language summary with proprioception, task id, and commander-state
metadata before the scalar value head. Do not reduce the VLA features to a fixed mean-pooled vector
before value training, and do not drop proprioception. Contact-heavy tasks need state and gripper
information.

Training targets:

- MC return-to-go from synthesized rewards.

Takeover risk is not a learned head in Stage A. Use `is_takeover_risk` deterministically for reward
shaping, actor-chunk exclusion, diagnostics, and future Stage B labels.

Validation gates before actor training:

- Held-out successful episodes should have higher values than held-out failed episodes.
- Values in successful episodes should trend upward toward terminal success after normalization.
- Deterministic takeover-risk windows should be present before `teleop` takeovers and absent in
  pure autonomous successes.
- Per-task value distributions must not collapse to a constant.
- The effective sample size of generated AWR weights must remain healthy.

If these checks fail, do not run expensive actor training. Fall back to static weighted SFT.

## 6. Stage A: `pi05awr_*`

Build an AWR index with the same key schema as the static weighted index:

```text
episode_index, frame_index, sample_weight, source_name, commander_state, ...
```

For each valid actor chunk start:

```text
A_t = sum_{k=0}^{N-1} r_{t+k} + V(s_{t+N}) - V(s_t)
A_t = robust_standardize_per_task_and_source(A_t)
awr = clip(exp(A_t / beta), awr_min, awr_max)
sample_weight = base_actor_weight(source, commander_state, takeover_region) * awr
```

Recommended defaults:

```text
beta: 1.0
awr_min: 0.25
awr_max: 3.0
pre_takeover actor weight: 0.0
failure-data actor weight: 0.0 for first Stage A runs
```

Use blending for the first run:

```text
sample_weight = (1 - rho) * static_weight + rho * awr_weight
rho in {0.25, 0.5, 1.0}
```

Start with cap or hanoi, not battery. Battery is already strong, so keep it close to static until
we know the value model is useful.

### Current Stage A Implementation Defaults

The current codebase implements the Stage A recipe with these concrete defaults.

Takeover and actor filtering:

```text
drop modes:
  pre_teleop
  restore
  align

takeover target modes:
  pre_teleop
  teleop

risk_window_frames:
  60

valid actor chunk:
  whole 50-frame action horizon must stay in one commander-state segment
  chunk must not overlap drop modes
  chunk must not overlap takeover-risk mask
  max absolute per-step action jump must be <= 0.2
  enough future frames must exist for the full action horizon
```

Actor base weights:

```text
expert-data:
  inference or teleop-like active modes: 1.0

success-and-hil-data:
  inference: 0.7
  teleop: 2.0
  pre_teleop: 0.0
  restore: 0.0
  align: 0.0
  inference risk window before takeover: 0.0

failure-data:
  default: 0.0
  configurable via --failure-actor-weight in build_takeover_awr_index.py
```

Value-target reward synthesis:

```text
step_penalty: -1.0
failure_terminal_penalty: -50.0
takeover_risk_penalty: -2.0
teleop_bonus: 0.0

successful terminal active frame reward:
  0.0

failed terminal active frame reward:
  -50.0

takeover-risk inference frames:
  step_penalty + takeover_risk_penalty = -3.0

pre_teleop frames:
  0.0 base drop-mode reward + takeover_risk_penalty = -2.0

return normalization:
  reward_target = reward / max(abs(return_to_go))
  value_target = return_to_go / max(abs(return_to_go))
  current implementation scales globally per target file
```

Feature extraction and value model:

```text
feature_key: pi05_prefix_tokens
request file: value-feature-requests/<task>.parquet
  includes valid actor starts, their N-step next states, and a critic-training frame subsample
  default actor_stride: 10
  default critic_stride: 30
feature format: per-episode .npz with frame_index [requested_frames], features [requested_frames, tokens, dim], and mask [requested_frames, tokens]
extract batch_size: 32

value model:
  frozen pi0.5 prefix token features
  learned query cross-attention aggregator
  hidden_dim: 128
  query_count: 8
  attention_heads: 8
  batch_size: 4096
  epochs: 20
  lr: 1e-3
  weight_decay: 1e-4
  val_fraction: 0.2
  seed: 42
  missing feature files fail by default
```

AWR index generation:

```text
action_horizon: 50
advantage_horizon: 50
max_action_jump: 0.2
beta: 1.0
awr_min: 0.25
awr_max: 3.0
rho default in script: 1.0
first-run rho used in docs/README examples: 0.25
reward_column: reward_target
value_column: value_pred if present, otherwise value_target

advantage:
  sum reward_target over [t, t + advantage_horizon)
  + V(s_{t + advantage_horizon})
  - V(s_t)

standardization:
  robust median/MAD within task if task_index/task_id exists, then source_name, commander_state
```

## 7. Stage B: `pi05ac_*` After Validation

Only implement Stage B after Stage A produces sensible value metrics and at least one promising
training run.

Stage B follows RECAP-style advantage conditioning:

- append a short token string to the prompt, e.g. `Advantage: positive` or `Advantage: negative`;
- randomly drop the token during training so the model learns conditional and unconditional modes;
- train the same flow-matching loss;
- at inference, condition on positive and optionally use conservative CFG.

Labels:

```text
forced positive:
  expert-data
  HIL teleop frames
  high-advantage autonomous success chunks

negative:
  failure-data
  pre_teleop
  inference risk window before takeover
  low-advantage autonomous chunks
```

Use conservative CFG first:

```text
cfg_beta: 1.2 to 1.8 initially
avoid high beta unless sim/video inspection shows stable motions
```

The prompt/token budget must be checked. If the added text causes prompt truncation or interferes
with task conditioning, Stage B should be paused.

## 8. What Not To Do

- Do not use `subtask` labels; they are `TODO`.
- Do not train a method that requires evaluator progress labels.
- Do not imitate `failure-data` as positive actor data in the first AWR implementation.
- Do not include `restore`, `align`, or `pre_teleop` in actor chunks.
- Do not let chunks cross takeover boundaries.
- Do not use offline TD/Q as the first RL method.
- Do not make CFG the primary submission before AWR/value validation.

## 9. Implementation Entry Points

Current branch assets that can be reused:

- `scripts/merge_lerobot.py`: merge leaves and preserve provenance.
- `src/openpi/training/challenge_weighting.py`: commander-state segmentation and chunk filters.
- `scripts/build_challenge_index.py`: static index builder to extend or wrap.
- `src/openpi/training/weighted_data_loader.py`: frame-keyed sample weights.
- `scripts/train_weighted.py`: weighted flow loss.
- `src/openpi/training/config.py`: add new `pi05awr_*` and later `pi05ac_*` configs.

New components needed for Stage A:

- reward/return synthesizer from provenance + commander states;
- takeover-region annotator;
- value dataset builder;
- value model trainer;
- offline value validation report;
- AWR index writer that emits `sample_weight`;
- synthetic LeRobot fixtures for local tests, because the full dataset is cluster-only.

New components needed only for Stage B:

- advantage-token prompt transform;
- token dropout during training;
- conditional/unconditional inference path;
- CFG sampler support for pi0.5 flow inference.

Stage A cluster command sequence for one task:

```bash
uv run python scripts/build_takeover_value_targets.py \
  --task-root "$CHALLENGE_ROOT/weighted-datasets/seal-water-bottle-cap-weighted-hil" \
  --output "$CHALLENGE_ROOT/value-targets/seal-water-bottle-cap.parquet"

uv run python scripts/train_takeover_value_model.py \
  --targets "$CHALLENGE_ROOT/value-targets/seal-water-bottle-cap.parquet" \
  --predictions "$CHALLENGE_ROOT/value-predictions/seal-water-bottle-cap.parquet" \
  --checkpoint "$CHALLENGE_ROOT/value-checkpoints/seal-water-bottle-cap.pt"

uv run python scripts/build_takeover_awr_index.py \
  --task-root "$CHALLENGE_ROOT/weighted-datasets/seal-water-bottle-cap-weighted-hil" \
  --value-targets "$CHALLENGE_ROOT/value-predictions/seal-water-bottle-cap.parquet" \
  --output "$CHALLENGE_ROOT/indexes/seal-water-bottle-cap-takeover-awr.parquet" \
  --rho 0.25

bash train_weighted.sh pi05awr_seal-water-bottle-cap_hil
```

## 10. Experiment Ladder

Run in this order:

1. Reproduce `main` expert-only baseline for one task on the cluster.
2. Run current static weighted SFT (`pi05w_*`) as the branch baseline.
3. Train and validate value model on one hard task (`seal-water-bottle-cap` or
   `tower-of-hanoi-game`).
4. Run `pi05awr_*` with `rho=0.25`, then `rho=0.5` or `1.0` if metrics are stable.
5. Repeat on the second hard task.
6. Keep battery near static unless AWR clearly helps.
7. Implement `pi05ac_*` only after Stage A validation.
8. Build specialist submissions first; multi-task generalist later with per-task normalization.

## 11. Related Work Grounding

- **pi0.6 / RECAP:** MC value from sparse success labels, advantage conditioning, CFG, failures
  as negative data, and human corrections forced positive. We use its value/AWR-compatible core
  first and defer full conditioning/CFG.
- **IG-RFT:** AWR on flow-matching losses and stop-gradient critic design are directly relevant.
  Its dense subtask rewards are not directly available here because the challenge dataset does not
  provide usable subtask labels.
- **ConRFT:** Offline Q learning alone is risky with small demonstration coverage; this supports
  avoiding TD/Q as the first Phase 1 method.
- **pi_RL / RLToken:** Useful Phase 2 directions if online interaction opens, but out of scope for
  Phase 1 offline training.
