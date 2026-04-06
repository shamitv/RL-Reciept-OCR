# Plan: RL Inference Path

## Status

- Current state: planned
- Current implementation gap:
  - `inference.py` only supports the heuristic agent
  - `training/train_ppo.py` is still a placeholder
  - there is no checkpoint-backed learned policy inference path yet

## Summary

- Extend the inference flow so the project can run a learned RL policy locally, not just the heuristic baseline.
- Keep `inference.py` as the main evaluation entrypoint, but add a real RL-backed agent mode alongside `heuristic`.
- Introduce a small policy-runtime layer that can load a trained checkpoint, encode observations, and return a valid `ReceiptAction`.
- Preserve the current heuristic mode as the default until a trained checkpoint exists.

## Implementation Changes

- Add a reusable RL policy inference module, likely under `training/` or `env/`, with responsibilities:
  - load model checkpoints from disk
  - convert `ReceiptObservation` into policy input features
  - run the policy forward pass
  - decode policy output into a valid `ReceiptAction`
  - apply action masking so invalid actions are not emitted
- Update `inference.py` to support at least:
  - `--agent heuristic`
  - `--agent ppo`
- Add checkpoint configuration flags such as:
  - `--checkpoint`
  - optionally `--device`
- Keep episode execution and reporting shared between heuristic and PPO modes so metrics remain directly comparable.
- Ensure the PPO path uses the same task loop, reward accounting, logging, and summary format as the heuristic path.
- Add graceful failure behavior when PPO mode is selected without a valid checkpoint:
  - fail fast with a clear error
  - do not silently fall back to heuristic mode

## Interfaces

- `inference.py` remains the main local entrypoint.
- Proposed CLI shape:
  - `python inference.py --agent heuristic`
  - `python inference.py --agent ppo --checkpoint checkpoints/policy.pt`
  - `python inference.py --agent ppo --checkpoint checkpoints/policy.pt --task hard --episodes 3`
- Keep existing output formats:
  - text summary
  - JSON summary
  - optional verbose step logs
- The PPO runtime should return the same action model used everywhere else:
  - `ReceiptAction`

## Policy Runtime Requirements

- Observation encoding must include enough structured information for the policy to act safely, such as:
  - difficulty
  - visible region statistics
  - candidate availability
  - current draft completion
  - validation state
  - remaining budget
  - previous action context if needed
- Action decoding must support the current typed action space rather than raw text generation.
- Invalid action masking should prevent obviously impossible choices, for example:
  - selecting a candidate before candidates exist
  - inspecting a non-visible or nonexistent region
  - submitting malformed action payloads

## Checkpoint Contract

- Define one explicit checkpoint format so inference and training agree.
- The checkpoint should include at minimum:
  - model weights
  - policy architecture metadata
  - observation encoder metadata
  - action-head metadata if needed for decoding
- Store enough metadata to reject incompatible checkpoints cleanly.

## Test Plan

- Add unit tests for observation encoding and action decoding.
- Add tests for invalid-action masking behavior.
- Add inference tests that mock a PPO policy and verify:
  - `--agent ppo` routes through the policy runtime
  - PPO mode requires a checkpoint
  - output summaries keep the same schema as heuristic mode
- Add one integration-style test that loads a minimal fake checkpoint and completes an episode through the PPO code path.
- Keep training itself out of scope for this phase; the goal here is learned-policy inference, not PPO optimization.

## Assumptions

- The learned policy will be an external trainable module, not the base LLM.
- PPO is the intended first learned-policy mode; additional learned agents can reuse the same inference scaffolding later.
- Heuristic mode remains the default until real checkpoints are available and stable.
- This plan covers running RL inference locally; full PPO training implementation is a related but separate follow-on task.
