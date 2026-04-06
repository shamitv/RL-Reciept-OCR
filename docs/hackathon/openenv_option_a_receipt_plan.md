# OpenEnv Receipt Understanding — Option A Implementation Plan

## Goal

Build an **OpenEnv environment** plus an **agent** for extracting structured data from receipt images in the **ICDAR 2019 SROIE** dataset.

This version follows **Option A**:

- the **base LLM remains frozen**
- a separate **trainable RL policy** learns how to act in the environment
- the RL policy decides **what to inspect, what to edit, when to validate, and when to submit**
- the LLM, if used at all, is only a helper for limited interpretation or reranking and is **not** the object being trained

The environment should be suitable for the OpenEnv hackathon style requirements:

- typed models
- `reset()` / `step()` / `state()` API
- at least 3 tasks
- deterministic graders with scores in `[0,1]`
- meaningful reward shaping
- reproducible baseline inference script
- deployable with Docker / HF Space

---

# 1. High-level design

## 1.1 What the system is

This is a **sequential document extraction environment**.

The agent does **not** receive the final answer directly.
It interacts over multiple steps.

At each step, the agent may:

- inspect OCR evidence
- inspect nearby regions
- query field candidates
- write or revise a draft field
- validate its current draft
- submit the final extraction

The episode ends when:

- the agent calls `submit()`, or
- the maximum step budget is exhausted

The environment then grades the extracted fields against the gold labels.

## 1.2 Target schema

Each receipt must be parsed into exactly these 4 fields:

- `company`
- `date`
- `address`
- `total`

## 1.3 Why this is a good RL environment

This is not just classification.
It is a **decision process**.

The agent must learn:

- which action is useful now
- which field is most uncertain
- when more evidence is needed
- when the draft is good enough to submit
- how to avoid wasting steps

That makes it appropriate for PPO or similar policy learning.

---

# 2. Option A architecture

## 2.1 Core principle

**The LLM does not learn.**
The **policy learns**.

There are 4 layers:

### Layer A — Environment
Implements receipt extraction task logic.

### Layer B — Deterministic helpers
Includes:
- OCR region indexing
- candidate generation
- normalization
- grading
- reward shaping

### Layer C — Frozen LLM helper (optional)
Used only for limited support tasks such as:
- reranking ambiguous candidates
- deciding whether two OCR spans likely belong together
- summarizing a small local ambiguity

This helper is optional and must not be required for correctness.

### Layer D — Trainable policy
A PPO policy network consumes structured state and outputs the next action.

This is the only trainable component in Option A.

## 2.2 What is learned

The RL policy learns:

- exploration order
- evidence-gathering strategy
- edit strategy
- validation timing
- stopping / submission behavior

It does **not** learn OCR or free-form text generation.

---

# 3. Environment specification

## 3.1 Episode definition

One episode = one receipt sample + one task variant.

The environment loads:

- receipt image
- OCR words and bounding boxes
- gold field labels
- task difficulty rules
- optional corruption / perturbation config

## 3.2 Hidden state

The environment internally stores:

- `sample_id`
- `difficulty`
- `image_path` or `image_id`
- `gold_fields`
- `all_ocr_regions`
- `revealed_region_ids`
- `current_draft`
- `history`
- `step_index`
- `remaining_budget`
- `done`
- `last_error`
- `cumulative_reward`
- reward bookkeeping such as best-so-far field quality

## 3.3 Observable state

The agent should only observe:

- task instructions
- difficulty
- currently revealed OCR evidence
- candidate lists already requested
- current draft
- validation feedback
- action history summary
- remaining budget
- whether submit is allowed

The environment must **not** reveal:

- gold labels
- unrevealed OCR regions
- hidden reward internals
- direct oracle correctness flags

---

# 4. Tasks

Implement **three tasks** first.

## 4.1 Task 1 — Easy: Clean OCR Extraction

### Description
The agent receives a relatively clean view of OCR and strong candidate lists.

### Intended challenge
Learn the basic workflow:
- inspect
- query
- set fields
- validate
- submit

### Configuration
- high OCR cleanliness
- low distractor rate
- generous action budget
- top header / amount regions easy to discover

### What success means
The agent can reliably fill all four fields with few steps.

## 4.2 Task 2 — Medium: Noisy OCR with Distractors

### Description
OCR is noisier and there are more distractor candidates.

### Intended challenge
The agent must reason about:
- similar-looking numbers
- split address lines
- multiple date-like strings
- header ambiguity

### Configuration
- moderate OCR corruption
- moderate candidate ranking noise
- moderate action budget
- more need for neighbor inspection and normalization

### What success means
The agent learns to gather more targeted evidence before locking fields.

## 4.3 Task 3 — Hard: Budgeted Extraction Under Ambiguity

### Description
The agent operates under tighter step budget and weaker candidate support.

### Intended challenge
The agent must learn:
- efficient exploration
- selective validation
- careful stopping behavior
- avoiding redundant actions

### Configuration
- stronger OCR noise
- weaker candidate ranking
- tighter action budget
- more distractor numeric fields
- some obvious regions hidden until asked for

### What success means
The agent improves score per step and stops wasting actions.

---

# 5. Action space

Use a **typed, structured action space**.

Do not let the RL policy emit arbitrary text as the main action.
Instead, use a factored action model.

## 5.1 Supported action types

### 1. `view_receipt`
Purpose:
- initialize visible metadata
- reveal high-level receipt info

### 2. `list_text_regions`
Arguments:
- `window`: one of `all`, `top`, `middle`, `bottom`

Purpose:
- reveal OCR regions in a chosen window

### 3. `inspect_bbox`
Arguments:
- `bbox_id`

Purpose:
- inspect one OCR region in more detail

### 4. `inspect_neighbors`
Arguments:
- `bbox_id`
- `radius_bucket`

Purpose:
- reveal nearby OCR regions
- especially useful for address and company blocks

### 5. `query_candidates`
Arguments:
- `field`: `company`, `date`, `address`, `total`

Purpose:
- ask deterministic candidate generator for best field candidates from visible evidence

### 6. `set_field_from_candidate`
Arguments:
- `field`
- `candidate_id`

Purpose:
- copy a candidate into the current draft

### 7. `set_field_manual`
Arguments:
- `field`
- `value`
- `evidence_ids`

Purpose:
- optional escape hatch when using a frozen LLM helper or manually constructed merge

Use sparingly in training.

### 8. `merge_spans`
Arguments:
- `field`
- `span_ids`
- `join_mode`

Purpose:
- combine multiple OCR regions into one field candidate
- most relevant for `address`

### 9. `normalize_field`
Arguments:
- `field`
- `mode`

Purpose:
- apply deterministic normalization like date parsing or amount formatting

### 10. `check_total_consistency`
Purpose:
- validate whether current `total` is supported by visible numeric evidence

### 11. `check_date_format`
Purpose:
- validate whether current `date` is plausibly normalized

### 12. `clear_field`
Arguments:
- `field`

Purpose:
- remove a bad guess

### 13. `submit`
Purpose:
- end the episode and trigger final grading

## 5.2 Invalid actions

These should be rejected or penalized:

- selecting a non-existent `bbox_id`
- selecting a candidate before querying that field
- merging spans not yet revealed
- submitting malformed payloads
- repeatedly asking for the same exact evidence without reason

---

# 6. Typed models

The actual code can vary, but a coding agent should implement models equivalent to the following.

## 6.1 OCR region model

```python
class OCRRegion(BaseModel):
    region_id: str
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float | None = None
    revealed: bool = True
```

## 6.2 Candidate model

```python
class FieldCandidate(BaseModel):
    candidate_id: str
    field: Literal["company", "date", "address", "total"]
    value: str
    evidence_ids: list[str]
    heuristic_score: float
```

## 6.3 Draft model

```python
class ReceiptDraft(BaseModel):
    company: str | None = None
    date: str | None = None
    address: str | None = None
    total: str | None = None
```

## 6.4 Action model

```python
class ReceiptAction(BaseModel):
    action_type: Literal[
        "view_receipt",
        "list_text_regions",
        "inspect_bbox",
        "inspect_neighbors",
        "query_candidates",
        "set_field_from_candidate",
        "set_field_manual",
        "merge_spans",
        "normalize_field",
        "check_total_consistency",
        "check_date_format",
        "clear_field",
        "submit",
    ]
    field: str | None = None
    window: str | None = None
    bbox_id: str | None = None
    radius_bucket: int | None = None
    candidate_id: str | None = None
    span_ids: list[str] | None = None
    join_mode: str | None = None
    mode: str | None = None
    value: str | None = None
    evidence_ids: list[str] | None = None
```

## 6.5 Observation model

```python
class ReceiptObservation(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    instruction: str
    image_ref: str | None
    visible_regions: list[OCRRegion]
    candidate_lists: dict[str, list[FieldCandidate]]
    current_draft: ReceiptDraft
    validation_feedback: list[str]
    last_action_result: str
    remaining_budget: int
    step_index: int
    terminal_allowed: bool
```

## 6.6 State model

```python
class ReceiptState(BaseModel):
    sample_id: str
    difficulty: str
    current_draft: ReceiptDraft
    revealed_region_ids: list[str]
    history: list[dict]
    step_index: int
    remaining_budget: int
    cumulative_reward: float
    done: bool
    last_error: str | None = None
```

---

# 7. Grader design

The grader must be deterministic.

## 7.1 Field scoring principles

### Company
Use normalized token-level matching.

### Date
Use exact match after date normalization.

### Address
Use token-level F1 after whitespace and punctuation normalization.

### Total
Use exact numeric match after amount normalization.

## 7.2 Recommended normalization rules

### Text normalization
- uppercase
- trim whitespace
- collapse repeated spaces
- remove low-value punctuation where appropriate

### Date normalization
Supported examples:
- `2019-03-25`
- `25/03/2019`
- `25-03-19`
- `03/25/2019` if it appears in OCR

Normalize to one canonical form like `YYYY-MM-DD`.

### Amount normalization
Normalize to two decimal places.
Examples:
- `31`
- `31.0`
- `31.00`

All become `31.00`.

### Address normalization
- trim and collapse spaces
- normalize line breaks to single spaces
- optionally remove duplicated punctuation

## 7.3 Per-field scores

```python
def company_score(pred, gold):
    return token_f1(normalize_text(pred), normalize_text(gold))

def date_score(pred, gold):
    return 1.0 if normalize_date(pred) == normalize_date(gold) else 0.0

def address_score(pred, gold):
    return token_f1(normalize_address(pred), normalize_address(gold))

def total_score(pred, gold):
    return 1.0 if normalize_amount(pred) == normalize_amount(gold) else 0.0
```

## 7.4 Overall score

Recommended weighting:

- company: `0.20`
- date: `0.20`
- address: `0.25`
- total: `0.35`

```python
def final_score(pred, gold):
    return (
        0.20 * company_score(pred.company, gold.company)
        + 0.20 * date_score(pred.date, gold.date)
        + 0.25 * address_score(pred.address, gold.address)
        + 0.35 * total_score(pred.total, gold.total)
    )
```

Clamp final score to `[0,1]`.

## 7.5 Success threshold

Recommended:

- `success = final_score >= 0.85`

---

# 8. Reward shaping

Reward shaping is critical.

The final reward should not be only binary.
The environment should provide useful learning signal across the trajectory.

## 8.1 Dense positive rewards

Suggested rewards:

- `+0.02` reveal new useful evidence
- `+0.03` fill a previously blank field with a plausible candidate
- `+0.02` improve normalized similarity of a field
- `+0.03` improve `total` plausibility via consistency check
- `+0.01` correctly merge address spans
- `+0.01` discover top-ranked valid date region

## 8.2 Penalties

Suggested penalties:

- `-0.01` repeated useless action
- `-0.02` invalid action payload
- `-0.02` set field with no supporting evidence
- `-0.03` overwrite a better field with a worse value
- `-0.01` per step after soft threshold
- `-0.05` submit too early with many blank fields

## 8.3 Terminal reward

At `submit()`:

- add `final_score` as terminal reward bonus

## 8.4 Important design rule

Reward **improvement deltas**, not absolute draft quality every step.

Maintain best-so-far field quality for each field.
Only reward improvement beyond the prior best.

This avoids reward farming from repeatedly re-scoring the same value.

---

# 9. Candidate generation

Candidate generation should be deterministic and not too powerful.
If it is too strong, PPO becomes unnecessary.

## 9.1 Company candidates

Use heuristics like:
- top-of-receipt preference
- large header block preference
- non-numeric text preference
- exclude obvious date / total / address regions

## 9.2 Date candidates

Use regex and OCR patterns for:
- `DD/MM/YYYY`
- `DD-MM-YYYY`
- `YYYY-MM-DD`
- compact date-like strings

## 9.3 Total candidates

Use numeric pattern extraction and ranking heuristics:
- amount near keywords like `TOTAL`, `AMOUNT`, `BALANCE`
- larger amount-like values
- lower-page preference if common in dataset

## 9.4 Address candidates

Use clustered line groups:
- nearby OCR lines
- address-like tokens
- multi-line merges
- exclude obvious totals or dates

## 9.5 Candidate generator constraints

The generator may help, but it must not be an oracle.
Especially in medium/hard tasks:
- add distractors
- weaken ranking
- sometimes omit the best candidate from the top-1 slot

---

# 10. PPO policy design

## 10.1 Key principle

PPO should act over **structured state**, not raw image pixels.

The environment and helper modules convert raw receipt data into a compact feature representation.

## 10.2 State vector contents

The policy input should include the following categories.

### A. Episode progress features
- difficulty one-hot
- step index
- remaining budget
- cumulative reward
- repeated action count
- invalid action count

### B. Draft features per field
For each field:
- blank flag
- edit count
- current heuristic confidence
- evidence count
- length bucket
- recently improved flag

### C. Evidence features
- number of visible OCR regions
- number of header-like regions visible
- number of amount-like regions visible
- number of date-like regions visible
- top candidate score per field
- ambiguity / entropy estimate per field

### D. Validation features
- total consistency status
- date validity status
- address completeness hint
- submit readiness estimate

### E. History features
- previous action type
- previous target field
- previous reward
- previous improvement flag

## 10.3 Recommended model

Start with a small MLP.

Example:
- input: structured feature vector
- hidden layers: 2 or 3 layers
- hidden size: 256 or 512
- outputs:
  - action type logits
  - field logits
  - argument logits
  - value head

This is simpler and more reliable than starting with a transformer.

## 10.4 Factorized action heads

Use separate output heads:

- `action_type_head`
- `field_head`
- `bbox_head`
- `candidate_head`
- `span_merge_head`
- `normalize_mode_head`
- `value_head`

Only use the relevant head outputs for the chosen action.

## 10.5 Invalid action masking

Mask impossible actions before sampling.

Examples:
- cannot `set_field_from_candidate` if no candidates exist
- cannot `inspect_bbox` if no regions revealed
- cannot `merge_spans` if no spans are visible

This will make PPO significantly more stable.

---

# 11. Role of the frozen LLM

The frozen LLM is optional.

If used, it should support only narrow subproblems.

## 11.1 Acceptable uses

- rerank top 2–5 candidates for one field
- decide whether two nearby text spans should be merged
- suggest best merged address out of a short list

## 11.2 Unacceptable uses

Do not let the LLM:
- bypass the environment and directly see gold labels
- decide the full trajectory by hidden internal logic
- replace the PPO policy entirely

## 11.3 Recommendation

Build version 1 with **no LLM dependency** in the critical path.
Then add optional LLM reranking later if useful.

---

# 12. Training plan

## 12.1 Phase 0 — Environment first

Before any RL:

1. load SROIE samples
2. normalize gold labels
3. build OCR region store
4. implement candidate retrieval
5. implement grader
6. implement reward shaping
7. implement all 3 tasks
8. run deterministic scripted rollouts

Exit criterion:
- environment works end to end
- scores are deterministic
- no training yet

## 12.2 Phase 1 — Behavior cloning warm start

Create expert-ish trajectories using heuristics or scripted policies.

Example scripted sequence:
1. `view_receipt`
2. `list_text_regions(top)`
3. `query_candidates(company)`
4. `set_field_from_candidate(company, top1)`
5. `query_candidates(date)`
6. `set_field_from_candidate(date, top1)`
7. `list_text_regions(bottom)`
8. `query_candidates(total)`
9. `set_field_from_candidate(total, top1)`
10. `query_candidates(address)`
11. `merge_spans(...)`
12. `check_total_consistency`
13. `submit`

Use these trajectories to pretrain the policy.

Exit criterion:
- low invalid action rate
- policy can complete easy task reasonably

## 12.3 Phase 2 — PPO on easy task

Train only on easy episodes first.

Goals:
- learn valid action sequencing
- reduce wasted actions
- learn submit timing

Metrics:
- average reward rising
- episode length stabilizing
- easy task score above heuristic floor

## 12.4 Phase 3 — PPO on easy + medium curriculum

Suggested mix:
- 70% easy
- 30% medium initially

Later:
- 50% easy
- 50% medium

Goals:
- robustness to OCR noise
- better disambiguation
- better use of validation actions

## 12.5 Phase 4 — PPO on all 3 tasks

Suggested mix:
- 20% easy
- 30% medium
- 50% hard

Goals:
- efficient evidence gathering
- stop wasting steps
- improved score on hard task

## 12.6 Phase 5 — Freeze and evaluate

Freeze the policy.
Run evaluation on:
- validation split
- held-out test split
- fixed seeds for reproducibility

Compare against:
- heuristic baseline
- frozen non-RL planner baseline
- PPO policy

---

# 13. PPO algorithm details

The exact library is flexible, but the coding agent should implement conventional PPO.

## 13.1 PPO setup

Suggested defaults:
- horizon: 8–12 steps per episode
- parallel envs: 8–32
- rollout size: enough for stable updates
- gamma: `0.97` to `0.99`
- GAE lambda: around `0.95`
- clipped objective
- entropy bonus early, decay later
- value function baseline

## 13.2 Training outputs

Track:
- policy loss
- value loss
- entropy
- mean reward
- mean final score
- mean episode length
- invalid action rate
- hard-task score

## 13.3 Recommended stopping rules

Stop when:
- validation score plateaus
- hard-task score stops improving
- policy begins over-querying or overfitting

---

# 14. Data augmentation / corruption

SROIE is not a huge dataset, so controlled training-time perturbation is useful.

## 14.1 Allowed training-time perturbations

- OCR substitutions such as `0/O`, `1/I`, `5/S`
- missing punctuation
- split lines
- merged lines
- bbox jitter
- candidate ranking noise
- hidden obvious regions until explicitly requested
- varying action budgets

## 14.2 Rules

- keep perturbations deterministic under seed
- do not change evaluation gold labels
- do not let perturbation break grader correctness

---

# 15. Baselines

Implement at least these baselines.

## 15.1 Baseline A — Heuristic policy

A deterministic scripted agent.

Purpose:
- establish floor performance
- debug environment
- produce reproducible non-RL benchmark

## 15.2 Baseline B — Frozen planner baseline

A non-RL agent that uses fixed rules or optional frozen LLM helper but no learning.

Purpose:
- compare against PPO fairly

## 15.3 Baseline C — PPO policy

The trained Option A policy.

Purpose:
- show actual learned improvement

---

# 16. Repo structure

Recommended repo structure:

```text
openenv-receipt-extract/
├─ README.md
├─ openenv.yaml
├─ inference.py
├─ Dockerfile
├─ requirements.txt
├─ pyproject.toml
├─ env/
│  ├─ __init__.py
│  ├─ models.py
│  ├─ dataset.py
│  ├─ normalizers.py
│  ├─ candidate_retrieval.py
│  ├─ graders.py
│  ├─ rewards.py
│  ├─ tasks.py
│  ├─ environment.py
│  ├─ server.py
│  └─ utils.py
├─ training/
│  ├─ build_bc_data.py
│  ├─ train_bc.py
│  ├─ train_ppo.py
│  ├─ eval_policy.py
│  └─ configs/
├─ tests/
│  ├─ test_normalizers.py
│  ├─ test_graders.py
│  ├─ test_rewards.py
│  ├─ test_environment.py
│  └─ test_determinism.py
└─ scripts/
   ├─ prepare_sroie.py
   ├─ smoke_test.py
   └─ validate_local.sh
```

---

# 17. Module responsibilities

## 17.1 `dataset.py`
Responsibilities:
- load raw SROIE records
- map receipt images to OCR words and bboxes
- map receipt images to gold entity fields
- create stable region IDs
- split train / val / test
- generate difficulty-specific episode views

## 17.2 `normalizers.py`
Responsibilities:
- text normalization
- date parsing and canonicalization
- amount parsing and canonicalization
- address whitespace normalization

## 17.3 `candidate_retrieval.py`
Responsibilities:
- generate company/date/address/total candidates from visible OCR
- cluster address spans
- rank numeric amount candidates
- produce deterministic candidate IDs

## 17.4 `graders.py`
Responsibilities:
- per-field score calculation
- final weighted score
- deterministic success flag

## 17.5 `rewards.py`
Responsibilities:
- compute dense reward increments
- compare previous and current draft quality
- track best-so-far field quality
- calculate penalties for redundancy and invalid actions

## 17.6 `environment.py`
Responsibilities:
- implement `reset()`, `step()`, `state()`
- maintain episode state
- call reward logic
- call grader on submit

## 17.7 `tasks.py`
Responsibilities:
- define easy / medium / hard task configs
- action budgets
- corruption rules
- evidence visibility policy

## 17.8 `server.py`
Responsibilities:
- serve environment endpoints for OpenEnv-compatible interaction

## 17.9 `train_bc.py`
Responsibilities:
- train policy via imitation from scripted trajectories

## 17.10 `train_ppo.py`
Responsibilities:
- PPO rollout collection
- masked action sampling
- PPO updates
- checkpointing and evaluation

---

# 18. Environment pseudocode

## 18.1 `reset()`

```python
def reset(self, task_name: str | None = None, seed: int | None = None):
    self.rng = make_rng(seed)
    self.task = self.task_registry.sample(task_name, rng=self.rng)
    self.sample = self.dataset.sample(self.task.difficulty, rng=self.rng)

    self.hidden_state = build_episode_state(self.sample, self.task, self.rng)
    self.hidden_state.done = False
    self.hidden_state.step_index = 0
    self.hidden_state.remaining_budget = self.task.max_steps
    self.hidden_state.current_draft = empty_draft()
    self.hidden_state.history = []
    self.hidden_state.revealed_region_ids = []
    self.hidden_state.cumulative_reward = 0.0
    self.hidden_state.best_field_scores = init_best_scores()

    obs = build_observation(self.hidden_state, message="episode reset")
    return StepResult(observation=obs, reward=0.0, done=False, info={})
```

## 18.2 `step()`

```python
def step(self, action: ReceiptAction):
    if self.hidden_state.done:
        return terminal_error_result("episode already done")

    prev_draft = deepcopy(self.hidden_state.current_draft)
    prev_meta = snapshot_reward_meta(self.hidden_state)

    action_result = execute_action(self.hidden_state, action)

    self.hidden_state.step_index += 1
    self.hidden_state.remaining_budget -= 1

    if action.action_type == "submit":
        final = self.grader.grade(self.hidden_state.current_draft, self.hidden_state.gold_fields)
        reward = self.rewarder.compute_terminal_reward(prev_draft, self.hidden_state, final)
        self.hidden_state.done = True
        info = {
            "success": final.success,
            "final_score": final.score,
            "field_scores": final.field_scores,
        }
    else:
        reward = self.rewarder.compute_step_reward(
            prev_draft=prev_draft,
            current_state=self.hidden_state,
            action=action,
            prev_meta=prev_meta,
            action_result=action_result,
        )
        info = {}

    if self.hidden_state.remaining_budget <= 0 and not self.hidden_state.done:
        final = self.grader.grade(self.hidden_state.current_draft, self.hidden_state.gold_fields)
        reward += self.rewarder.compute_budget_exhaustion_reward(final)
        self.hidden_state.done = True
        info.update({
            "success": final.success,
            "final_score": final.score,
            "budget_exhausted": True,
        })

    self.hidden_state.cumulative_reward += reward
    obs = build_observation(self.hidden_state, message=action_result.message)

    return StepResult(
        observation=obs,
        reward=clip_reward(reward),
        done=self.hidden_state.done,
        info=info,
    )
```

## 18.3 `state()`

```python
def state(self):
    return build_state_model(self.hidden_state)
```

---

# 19. PPO training pseudocode

```python
policy = PolicyNetwork(...)
optimizer = Adam(policy.parameters(), lr=...)

envs = make_vectorized_envs(num_envs=N)

for update_idx in range(num_updates):
    rollout = []

    obs_batch = envs.reset()

    for t in range(horizon):
        state_vec = encode_obs_batch(obs_batch)
        masks = build_action_masks(obs_batch)

        action_dist, value = policy.forward(state_vec, masks=masks)
        action = action_dist.sample()
        logprob = action_dist.log_prob(action)

        next_obs_batch, reward_batch, done_batch, info_batch = envs.step(action)

        rollout.append({
            "obs": obs_batch,
            "action": action,
            "reward": reward_batch,
            "done": done_batch,
            "value": value,
            "logprob": logprob,
            "info": info_batch,
        })

        obs_batch = next_obs_batch

    advantages, returns = compute_gae(rollout, policy)

    for epoch in range(ppo_epochs):
        for minibatch in iterate_minibatches(rollout, advantages, returns):
            loss = ppo_loss(policy, minibatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    evaluate(policy, val_envs)
    save_checkpoint_if_best(policy)
```

---

# 20. Inference script behavior

The submission needs a deterministic baseline inference script.

`inference.py` should:

1. instantiate the environment
2. run the chosen policy per task
3. emit strict logs
4. return reproducible scores

## 20.1 Recommended inference modes

Support flags like:
- `--agent heuristic`
- `--agent frozen`
- `--agent ppo`

## 20.2 PPO inference behavior

At inference time:
- load trained checkpoint
- build observation features
- apply action masks
- choose greedy action or temperature 0 equivalent
- stop on submit or budget exhaustion

---

# 21. Testing checklist

A coding agent should implement these tests.

## 21.1 Normalizer tests
- amount normalization edge cases
- date normalization edge cases
- whitespace collapse behavior

## 21.2 Grader tests
- exact match cases
- partial address match cases
- malformed prediction cases
- score clamping to `[0,1]`

## 21.3 Reward tests
- field improvement reward
- repeated action penalty
- premature submit penalty
- terminal reward correctness

## 21.4 Environment tests
- reset returns valid initial observation
- step updates state correctly
- submit terminates episode
- invalid actions are handled deterministically

## 21.5 Determinism tests
- same seed + same action sequence => same observation/reward sequence

---

# 22. Evaluation metrics

Track all of these during training and final evaluation.

## 22.1 Primary metrics
- average final score per task
- overall weighted mean score
- full-record exact match rate

## 22.2 Per-field metrics
- company score
- date accuracy
- address F1
- total accuracy

## 22.3 Efficiency metrics
- average steps to submit
- invalid action rate
- repeated action rate
- reward per episode

## 22.4 RL-specific metrics
- policy entropy
- value loss
- PPO clipped fraction
- hard-task improvement over heuristic baseline

---

# 23. Risk register

## Risk 1 — Candidate generator too good
If the generator already solves the problem, PPO adds little value.

Mitigation:
- weaken ranking on medium/hard
- add distractors
- require targeted inspection

## Risk 2 — PPO learns to spam validation
Mitigation:
- cap validation reward
- add step cost
- penalize repeated checks

## Risk 3 — PPO submits too early
Mitigation:
- add premature submit penalty
- add auxiliary submit-readiness feature
- use imitation warm start

## Risk 4 — Overfitting to small dataset
Mitigation:
- training-time corruption
- strict validation split
- keep policy small

## Risk 5 — Runtime too slow
Mitigation:
- precompute OCR and region indices offline
- keep max steps low
- avoid expensive LLM calls in critical path

---

# 24. Delivery order

A coding agent should implement the project in this order.

## Milestone 1
- dataset loader
- normalizers
- graders

## Milestone 2
- easy task environment
- heuristic baseline
- reward shaping v1

## Milestone 3
- medium and hard tasks
- candidate retrieval
- action masks

## Milestone 4
- behavior cloning trajectories
- BC warm start

## Milestone 5
- PPO training
- evaluation loop
- checkpointing

## Milestone 6
- OpenEnv packaging
- `openenv.yaml`
- Dockerfile
- `inference.py`
- HF Space deployment

---

# 25. Definition of done

The Option A implementation is complete when all of the following are true:

1. Environment supports typed `reset()` / `step()` / `state()` interaction.
2. Three tasks exist: easy, medium, hard.
3. Grader is deterministic and returns `[0,1]` score.
4. Reward shaping gives useful signal before terminal step.
5. Heuristic baseline runs end to end.
6. PPO policy trains and improves at least one meaningful metric, ideally hard-task score or efficiency.
7. `inference.py` runs reproducibly.
8. Docker build works.
9. Environment can be served in OpenEnv-compatible form.
10. README documents action space, observation space, tasks, setup, and baseline results.

---

# 26. Recommended implementation notes for a coding agent

- Start with **no LLM dependency**.
- Make the environment deterministic before adding RL.
- Prefer **simple, auditable heuristics** for candidate generation.
- Keep the PPO action space **small and masked**.
- Use behavior cloning warm start to reduce invalid actions.
- Optimize hard-task score and action efficiency, not only easy-task accuracy.
- Avoid premature complexity like multimodal training or end-to-end OCR learning.
- Keep the runtime footprint small enough for hackathon infra.

---

# 27. Final one-paragraph summary

This project implements a receipt extraction OpenEnv environment over the ICDAR 2019 SROIE dataset, where an agent must sequentially gather evidence, update a structured draft, validate uncertain fields, and decide when to submit. The base LLM remains frozen; learning happens in a separate PPO policy trained on structured environment observations and shaped reward. The environment includes three difficulty levels, deterministic field graders, reproducible baselines, and a deployment-friendly package structure suitable for OpenEnv-style evaluation.
