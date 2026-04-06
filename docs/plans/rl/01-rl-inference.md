# Plan: RL Inference Path

## Status

- Current state: implemented
- Current implementation gap:
  - `training/train_ppo.py` is still a placeholder
  - learned-policy training is still separate from the implemented checkpoint-backed inference path

## Summary

- Extend the inference flow so the project can run a learned RL policy locally, not just the heuristic baseline.
- Keep `inference.py` as the main evaluation entrypoint, but add a real RL-backed agent mode alongside `heuristic`.
- Introduce a small policy-runtime layer that can load a trained checkpoint, encode observations, and return a valid `ReceiptAction`.
- Preserve the current heuristic mode as the default until a trained checkpoint exists.

## Module Layout

Create a new `agents/` top-level directory to separate agent logic from environment logic and training loops:

```
agents/
  __init__.py
  base.py          # Agent protocol (select_action interface)
  heuristic.py     # Existing heuristic_action logic, extracted from inference.py
  ppo.py           # PolicyRuntime: load checkpoint, encode obs, decode action
  encoding.py      # Observation encoder (shared between training and inference)
```

Rationale:
- `env/` stays focused on environment mechanics (state, transitions, rewards, grading).
- `training/` stays focused on training loops.
- `agents/` owns everything about action selection — the protocol, the heuristic baseline, and learned policies.
- `agents/encoding.py` is importable by both training and inference, ensuring the observation tensor format is defined in one place.

## Agent Interface

Define a minimal agent protocol so `run_episode` and `evaluate_tasks` are agent-agnostic:

```python
# agents/base.py
from typing import Protocol
from env.environment import ReceiptExtractionEnv
from env.models import ReceiptAction

class Agent(Protocol):
    name: str
    def select_action(self, env: ReceiptExtractionEnv) -> ReceiptAction: ...
```

Refactoring required in `inference.py`:
- Extract `heuristic_action()` and its helpers (`tried_windows`, `next_window_to_reveal`) into `agents/heuristic.py` as a `HeuristicAgent` class.
- Parameterize `run_episode(task, seed, agent, ...)` and `evaluate_tasks(tasks, seed, episodes, agent, ...)` to accept an `Agent` instance.
- Replace hardcoded `agent="heuristic"` in log messages and summary dicts with `agent.name`.
- Update `build_parser` to accept `choices=["heuristic", "ppo"]` and add `--checkpoint` / `--device` flags.
- In `main()`, construct the correct agent based on `args.agent`.

## Implementation Changes

- Create `agents/` directory with the module layout above.
- Update package discovery so installed builds include the new package:
  - in `pyproject.toml`, extend `[tool.setuptools.packages.find].include` to cover `agents*`
- Extract heuristic agent from `inference.py` into `agents/heuristic.py`.
- Implement `agents/encoding.py` with the observation encoder (see Observation Encoding below).
- Implement `agents/ppo.py` with the policy runtime (see Action Decoding and Policy Runtime Requirements below).
- Refactor `inference.py` to use the `Agent` protocol, parameterizing episode execution.
- Add checkpoint configuration flags:
  - `--checkpoint` (required for `--agent ppo`)
  - `--device` (optional, defaults to `cpu`)
- Keep episode execution and reporting shared between heuristic and PPO modes so metrics remain directly comparable.
- Ensure the PPO path uses the same task loop, reward accounting, logging, and summary format as the heuristic path.
- Add graceful failure behavior when PPO mode is selected without a valid checkpoint:
  - fail fast with a clear error
  - do not silently fall back to heuristic mode
- Supersede `training/eval_policy.py`: the updated `inference.py --agent ppo` replaces its purpose. Remove the placeholder or convert it to a thin wrapper that calls `inference.py` with PPO args for use during training loops.

## Interfaces

- `inference.py` remains the main local entrypoint.
- Proposed CLI shape:
  - `python inference.py --agent heuristic`
  - `python inference.py --agent ppo --checkpoint checkpoints/policy.pt`
  - `python inference.py --agent ppo --checkpoint checkpoints/policy.pt --task hard --episodes 3`
  - `python inference.py --agent ppo --checkpoint checkpoints/policy.pt --device cuda`
- Keep existing output formats:
  - text summary
  - JSON summary
  - optional verbose step logs
- The PPO runtime should return the same action model used everywhere else:
  - `ReceiptAction`

## Observation Encoding

Strategy: **fixed-width summary vector**.

`ReceiptObservation` contains variable-length fields (`visible_regions`, `candidate_lists`, `validation_feedback`) that cannot be fed directly to a fixed-architecture MLP. Rather than padding sequences (which adds complexity and fragility), encode a fixed-width summary vector that captures the decision-relevant statistics.

Proposed feature vector (all floats, total ~26 dims):

| Feature | Dims | Source |
|---|---|---|
| Difficulty one-hot | 3 | `obs.difficulty` ∈ {easy, medium, hard} |
| Step progress | 1 | `obs.step_index / task.max_steps` |
| Budget remaining ratio | 1 | `obs.remaining_budget / task.max_steps` |
| Num visible regions | 1 | `len(obs.visible_regions)` |
| Mean region confidence | 1 | `mean(r.confidence for r in obs.visible_regions)` |
| Region spatial coverage | 3 | fraction of regions in top/mid/bottom thirds |
| Candidates per field (4) | 4 | `len(obs.candidate_lists.get(field, []))` for each field |
| Top candidate score per field (4) | 4 | `max(c.heuristic_score)` per field, 0 if no candidates |
| Draft completion per field (4) | 4 | 1.0 if `getattr(obs.current_draft, field)` is set, else 0.0 |
| Validation feedback count | 1 | `len(obs.validation_feedback)` |
| Has positive validation | 1 | any "valid" or "matches" in feedback |
| Has negative validation | 1 | any "invalid" or "weakly" in feedback |
| Terminal allowed | 1 | `float(obs.terminal_allowed)` |

Total: ~26 dimensions. This is compact enough for a small MLP policy and captures all decision-relevant signals without requiring sequence modeling.

Implementation: `agents/encoding.py` exposes `encode_observation(obs: ReceiptObservation, task: TaskConfig) -> torch.Tensor` which returns a 1-D float tensor. The exact dimension count is stored in checkpoint metadata as `obs_dim`.

Design note:
- Keep the initial encoder derivable from `ReceiptObservation` + `TaskConfig` only.
- Do not include "last action type" in v1, because the current `ReceiptObservation` model does not expose it directly.
- If previous-action context is added later, treat it as an encoder-version change and bump `encoder_version`.

## Action Decoding

Strategy: **hierarchical heads**.

`ReceiptAction` is not a flat categorical — it's a structured action with an `action_type` enum (13 values) plus type-dependent parameters. A single softmax over action types is insufficient; the policy must also predict correct payloads.

Architecture:

1. **Shared trunk**: The observation vector passes through a small MLP (2–3 hidden layers) producing a shared representation.
2. **Action-type head**: A linear layer + softmax over the 13 action types, with an action mask applied before softmax to zero out invalid types.
3. **Parameter heads** (one per action type that requires parameters):
   - `list_text_regions` → window selector (categorical over available windows)
   - `query_candidates` → field selector (categorical over 4 fields)
   - `set_field_from_candidate` → field selector + candidate index
   - `inspect_bbox` → bbox index (categorical over visible region IDs)
   - `inspect_neighbors` → bbox index + radius bucket
   - `set_field_manual` → field selector + value (deferred; may be out of scope for initial PPO)
   - `merge_spans` → field + span selection (deferred; combinatorial)
   - `normalize_field` → field selector
   - `clear_field` → field selector

4. **Execution**: At inference, the model selects the action type first, then runs only the relevant parameter head to produce the full `ReceiptAction`.

Simplification for initial implementation:
- Actions requiring free-text output (`set_field_manual`) or combinatorial span selection (`merge_spans`) can be masked out initially. The policy can learn to use `set_field_from_candidate` and `normalize_field` instead.
- This reduces the parameter heads to simple categoricals.
- The initial PPO runtime therefore targets this explicit supported action subset:
  - `view_receipt`
  - `list_text_regions`
  - `inspect_bbox`
  - `inspect_neighbors`
  - `query_candidates`
  - `set_field_from_candidate`
  - `normalize_field`
  - `check_total_consistency`
  - `check_date_format`
  - `clear_field`
  - `submit`

The checkpoint stores the number and configuration of parameter heads in `param_heads`.

## Policy Runtime Requirements

- Observation encoding uses the fixed-width summary strategy defined above.
- Action decoding uses the hierarchical-head architecture defined above.
- Invalid action masking should prevent obviously impossible choices, for example:
  - selecting a candidate before candidates exist
  - inspecting a non-visible or nonexistent region
  - `query_candidates` for a field that already has candidates and a set draft value
  - `submit` when no fields are filled
  - submitting malformed action payloads
- The mask is computed from the current `ReceiptObservation` and applied to the action-type logits before softmax.
- Parameter-level masking (e.g., only allowing candidate IDs that exist) is applied to the corresponding parameter head.

## Checkpoint Contract

- Define one explicit checkpoint format so inference and training agree.
- The checkpoint should include at minimum:
  - `model_state_dict`: model weights
  - `obs_dim`: integer, observation vector dimension
  - `action_types`: list of supported PPO action type strings, in the order used by the action-type head
  - `param_heads`: dict mapping action types to their parameter head configs (num classes, etc.)
  - `architecture`: dict with hidden sizes, number of layers, activation function
  - `encoder_version`: string, version tag for the observation encoder (to detect incompatible encoding changes)
  - `training_metadata`: optional dict with training hyperparams, epoch, reward stats for provenance
- Store enough metadata to reject incompatible checkpoints cleanly:
  - On load, assert `obs_dim` matches the current encoder output.
  - On load, assert `action_types` matches the runtime's explicit supported PPO action subset for this version.
  - On mismatch, raise a clear `CheckpointIncompatibleError` with details.

## Dependencies

- `torch` is required for PPO inference but should be an **optional dependency** to avoid forcing a ~2GB install for heuristic-only usage.
- In `pyproject.toml`, add an optional extra:
  ```toml
  [project.optional-dependencies]
  ppo = ["torch>=2.0"]
  ```
- In `pyproject.toml`, keep package discovery aligned with the new module layout:
  ```toml
  [tool.setuptools.packages.find]
  include = ["env*", "server*", "agents*"]
  ```
- In `agents/ppo.py`, guard the torch import:
  ```python
  try:
      import torch
  except ImportError:
      raise ImportError(
          "torch is required for --agent ppo. "
          "Install with: pip install -e '.[ppo]'"
      )
  ```
- The heuristic agent path must remain fully functional without torch installed.

## Documentation Updates

- Update `docs/architecture/overall-system-architecture.md` to show the new `agents/` layer, the checkpoint-backed PPO inference path, and how `inference.py` selects between heuristic and learned agents.
- Update `docs/architecture/how-rl-is-applied.md` to describe the inference-time PPO runtime, the supported PPO action subset, and how the learned policy interacts with the existing environment loop.
- Update `docs/architecture/frameworks-and-libraries.md` to document the optional `torch` dependency, checkpoint loading expectations, and the fact that Gymnasium compatibility is still deferred to the training phase.
- Update `docs/architecture/README.md` and `docs/README.md` if needed so the architecture index still points readers to the most relevant RL and inference documentation after this work lands.
- Keep the architecture docs explicit about current state versus planned state until PPO inference is actually implemented.

## Gymnasium Compatibility

- The current `ReceiptExtractionEnv` uses a custom `reset`/`step` API with Pydantic models, not Gymnasium spaces.
- Standard RL libraries (SB3, CleanRL, RLlib) expect `gymnasium.Env` with `observation_space` and `action_space`.
- **Decision: defer Gymnasium wrapping to the training plan.** This plan focuses on inference only. The observation encoder and action decoder are designed as standalone modules that can be reused whether or not a Gymnasium wrapper exists.
- When the PPO training plan is written, it should introduce a `GymReceiptEnv` wrapper in `training/` that adapts the environment for whichever RL library is chosen. The encoder and decoder from `agents/encoding.py` will be shared.

## Test Plan

- Add unit tests for observation encoding:
  - Verify output tensor shape matches `obs_dim`.
  - Verify encoding is deterministic for the same observation.
  - Verify edge cases: zero regions, zero candidates, all fields filled, all fields empty.
- Add unit tests for action decoding:
  - Verify each action type produces a valid `ReceiptAction`.
  - Verify parameter heads produce values within valid ranges.
- Add tests for invalid-action masking behavior:
  - Verify masking blocks `set_field_from_candidate` when no candidates exist for any field.
  - Verify masking blocks `inspect_bbox` for non-revealed region IDs.
  - Verify masking permits `submit` only when at least one field is set.
  - Verify masking permits `view_receipt` only at step 0.
- Add inference tests that mock a PPO policy and verify:
  - `--agent ppo` routes through the policy runtime.
  - `--agent ppo` without `--checkpoint` fails with a clear error message.
  - Output summaries keep the same schema as heuristic mode.
- Add one integration-style test that loads a minimal fake checkpoint and completes an episode:
  - The fake checkpoint is a random-weight model saved with the exact checkpoint contract above (correct `obs_dim`, `action_types`, `param_heads`, `architecture`, `encoder_version`).
  - Use a fixed seed for determinism.
  - Assert: checkpoint loads without error.
  - Assert: episode runs to completion (done=True).
  - Assert: all emitted actions pass action-masking validation (no invalid actions produced).
  - Assert: output summary dict has the same keys as heuristic mode.
- Keep training itself out of scope for this phase; the goal here is learned-policy inference, not PPO optimization.

## Assumptions

- The learned policy will be an external trainable module (MLP), not the base LLM.
- PPO is the intended first learned-policy mode; additional learned agents can reuse the same `Agent` protocol and inference scaffolding later.
- Heuristic mode remains the default until real checkpoints are available and stable.
- This plan covers running RL inference locally; full PPO training implementation is a related but separate follow-on task.
- The environment action enum remains broader than the initial PPO-supported subset.
- `set_field_manual` and `merge_spans` are excluded from the initial PPO action space due to their free-form / combinatorial parameter requirements. They can be added later with more expressive parameter heads and a new checkpoint/runtime version.
- Gymnasium wrapping is deferred to the training plan.
