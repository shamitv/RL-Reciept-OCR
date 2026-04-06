# Frameworks And Libraries

## Overview

The project intentionally uses a small Python stack. Most of the domain logic is implemented in plain Python plus typed models rather than in a heavyweight ML framework.

## Core Runtime Stack

### Python

- Python `>=3.11`

Why it matters:

- modern typing support
- solid async/web ecosystem support
- good fit for FastAPI, Pydantic, and OpenAI-compatible clients

### Pydantic

- package: `pydantic`

Used for:

- typed action, observation, state, and result models
- structured environment payloads
- validation and serialization across the API boundary

Primary role in architecture:

- makes the OpenEnv-facing data contract explicit and testable

### FastAPI

- package: `fastapi`

Used for:

- OpenEnv endpoints: `/reset`, `/step`, `/state`
- eval endpoints under `/api/eval`
- serving the evaluation UI

Primary role in architecture:

- the main web/API framework for the project

### Uvicorn

- package: `uvicorn`

Used for:

- local development server
- running the FastAPI app in a lightweight ASGI server

### Jinja2

- package: `jinja2`

Used for:

- server-rendered evaluation UI templates

Primary role in architecture:

- keeps the UI simple without adding a separate frontend build toolchain

## Data And Normalization Utilities

### python-dateutil

- package: `python-dateutil`

Used for:

- parsing and normalizing date strings extracted from receipt annotations or model output

### Standard Library

Heavily used modules include:

- `json`
- `pathlib`
- `dataclasses`
- `statistics`
- `random`
- `hashlib`
- `logging`
- `datetime`

This is significant because much of the system’s actual logic is implemented in standard-library code rather than external frameworks.

## Model And Evaluation Integrations

### PyTorch

- package: optional extra `ppo = ["torch>=2.0"]`

Used for:

- loading learned-policy checkpoints for PPO inference
- encoding observations into tensors
- running the MLP policy trunk, action head, and parameter heads

Architectural note:

- PyTorch is optional because heuristic-only usage does not need it
- PPO training is still not implemented, but the inference runtime now depends on PyTorch when `--agent ppo` is used

### OpenAI Python Client

- package: `openai`

Used for:

- OpenAI-compatible extraction requests in the dataset eval flow
- OpenAI-compatible judge requests in the dataset eval flow

Architectural note:

- this is used as a client protocol, not as proof that only OpenAI-hosted models can be used
- the code accepts base URLs, so compatible providers can be used behind the same interface

### python-dotenv

- package: `python-dotenv`

Used for:

- loading `.env` configuration from disk at startup

Important configuration areas:

- dataset location
- extractor model and base URL
- judge model and base URL
- eval artifact location
- LLM cache location and TTL

### LLM Cache

- implemented locally in `env/llm_cache.py`

Used for:

- exact-match disk caching of OpenAI-compatible chat completion calls

Architectural role:

- reduces repeated eval latency and repeated model spend for identical requests

## OpenEnv And Packaging

### openenv-core

- package: `openenv-core`

Used for:

- validation and compatibility with the OpenEnv benchmark structure

Architectural role:

- anchors the project to the expected environment contract for the hackathon target

### setuptools / wheel

Used for:

- packaging and distribution metadata

### pytest

- package: `pytest`

Used for:

- unit and integration-style tests across the environment, grading, dataset loading, eval API, and eval pipeline

## What Is Not In The Runtime Stack Today

These are intentionally absent from the current checked-in implementation:

- LangGraph
- TensorFlow
- JAX
- Stable-Baselines3
- Ray RLlib
- a dedicated frontend framework such as React, Next.js, or Vue

That absence matters because it tells you the current codebase is:

- environment-first
- deterministic-first
- API-and-eval focused
- not yet shipping a real PPO training implementation

## Planned But Not Yet Implemented RL Stack

The architecture and plans describe a broader PPO-based learning stack, but the training code is still placeholder-only.

That means the likely future framework set will need to add at least:

- rollout collection and optimization tooling
- a PPO training loop on top of the existing PyTorch-based inference runtime
- possibly a Gymnasium adapter or higher-level RL library if the training plan chooses one

Those training-specific libraries are not yet part of the project’s declared runtime dependencies.

## Practical Summary

Implemented stack today:

- Python
- Pydantic
- FastAPI
- Uvicorn
- Jinja2
- optional PyTorch for PPO inference
- OpenAI client
- python-dotenv
- python-dateutil
- openenv-core
- pytest

Planned stack for learned-policy work:

- PPO training framework and supporting optimization tooling, still to be chosen and implemented
