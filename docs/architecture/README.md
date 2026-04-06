# Architecture Docs

This section describes how OpenEnv Receipt Understanding is organized today and how the planned RL workflow fits into it.

## Documents

- [Overall System Architecture](overall-system-architecture.md)
  - end-to-end system view
  - runtime components
  - main data and control flows

- [How RL Is Applied](how-rl-is-applied.md)
  - RL formulation of the receipt extraction task
  - trainable vs. frozen components
  - what is implemented today vs. what is still planned

- [Frameworks And Libraries](frameworks-and-libraries.md)
  - core Python/runtime dependencies
  - web/API stack
  - model/eval integrations
  - testing and packaging tools

## Current State

- The environment, API server, deterministic grading, eval pipeline, eval UI, and PPO inference runtime are implemented.
- PPO and BC training code are still placeholders in `training/`.
