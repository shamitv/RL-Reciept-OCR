# How RL Is Applied

## Short Answer

RL is applied to the policy that interacts with the receipt-extraction environment, not to the base LLM.

The intended learning agent is:

- a policy that reads environment observations
- chooses the next action
- gets reward feedback
- improves through PPO updates over time

## RL Formulation

The project treats receipt extraction as a sequential decision problem instead of a single one-shot prediction.

### State / Observation

The policy can observe structured environment output such as:

- task and difficulty
- visible OCR regions
- candidate lists
- current draft
- validation feedback
- remaining step budget
- step index

### Action

The policy chooses from typed actions, including:

- reveal the receipt or a window of OCR text
- inspect a region or nearby regions
- query candidates for one field
- set or clear a field
- normalize or validate a field
- submit the final draft

### Reward

The reward function provides:

- positive signal when the draft improves
- penalties for wasteful or invalid actions
- a terminal reward based on the final deterministic grade

### Objective

The policy should maximize task completion quality under the environment’s step budget and uncertainty.

## What Actually Learns

The trainable component is the policy module.

Conceptually:

```text
observation -> policy network -> action
                    ^
                    |
             updated by PPO from reward
```

That policy is meant to learn:

- what action to take next
- which field to focus on
- whether to inspect more OCR evidence
- which candidate to select
- when validation is worthwhile
- when the draft is good enough to submit

This is a real learning agent because later behavior is intended to improve based on prior reward.

## Trainable vs. Frozen Components

### Trainable

- the external policy network trained with PPO

### Frozen

- the base LLM, if used as a helper
- the OCR evidence already present in the dataset
- the deterministic graders
- the reward logic
- the environment mechanics

## What RL Does Not Cover

The current design does not use RL to:

- fine-tune the base LLM
- learn OCR from pixels
- directly produce end-to-end free-form text generations as the main control mechanism

Instead, RL operates over a structured action space on top of a deterministic environment.

## Why This Fits The Project

Receipt extraction in this environment is not just “predict four fields.”

The agent must decide:

- where to look
- what evidence is relevant
- whether the current evidence is sufficient
- when it is worth spending more steps
- when to stop

That makes the problem a good fit for RL because the challenge is partly about search strategy and decision sequencing, not only final-value prediction.

## Current Code Status

Implemented today:

- the environment, observation model, actions, rewards, and heuristic policy baseline

Not yet implemented:

- the PPO learner
- the behavior-cloning warm start
- the learned policy inference path

In other words, the codebase is currently RL-ready in problem formulation, but not yet RL-complete in training implementation.

## Planned Training Loop

The intended future loop is:

```mermaid
flowchart TD
    A["ReceiptExtractionEnv reset()"] --> B["Policy reads observation"]
    B --> C["Policy selects action"]
    C --> D["Environment step(action)"]
    D --> E["Observation + reward + done"]
    E --> F["Rollout buffer"]
    F --> G["PPO update"]
    G --> B
```

## Relationship To LLM Usage

If an LLM is used in the broader system, its role is helper-only in this RL framing:

- reranking
- interpretation
- evaluation-time extraction or judging

The LLM is not the policy that is being optimized by PPO.

So the architectural split is:

- RL learns the action policy
- deterministic code defines the environment and grading
- optional LLM calls assist specific tasks but remain frozen
