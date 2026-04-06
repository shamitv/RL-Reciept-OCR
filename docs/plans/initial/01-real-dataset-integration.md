# Plan: Real Dataset Integration

## Goal

Replace the embedded mock receipt samples with a deterministic, testable loader built on the prepared receipt dataset in the repository.

## Problem Statement

The current implementation demonstrates the environment API, but not the actual real-world task. The live environment pulls samples from hard-coded in-memory objects instead of reading prepared receipt artifacts from disk.

## Desired End State

After this plan is complete:

- the environment loads receipt samples from the project dataset directory
- sample selection remains deterministic under a seed
- malformed or incomplete records fail with clear diagnostics
- tests cover the data path without depending on the full dataset size
- mock samples are either removed or retained only as explicit test fixtures

## Scope

### In scope

- dataset discovery and parsing
- mapping dataset files into `ReceiptSample`
- deterministic split and sampling behavior
- validation helpers for required fields and OCR regions
- dataset-oriented tests

### Out of scope

- model training changes
- OpenAI policy prompting
- README polish beyond what is needed to explain dataset setup

## Assumptions To Verify

1. The prepared dataset has image files, OCR text regions, and field annotations that can be aligned by receipt ID.
2. The `dataset/Receipt dataset/` tree is the intended source of truth for the environment runtime.
3. The annotation JSON files contain enough information to build the four target fields: `company`, `date`, `address`, and `total`.
4. OCR bounding boxes are either already available or can be reconstructed from the prepared artifacts.

## Implementation Plan

### Step 1: Define the canonical on-disk contract

Tasks:

- inspect a representative set of annotation files and image references
- document which fields are mandatory and which are optional
- identify how receipt ID, OCR lines, and gold labels are linked
- decide whether the loader uses raw source files directly or a normalized prepared manifest

Output:

- a short schema note in code comments or a dedicated docstring near the loader

Acceptance criteria:

- a new engineer can understand how one dataset item becomes one `ReceiptSample`

### Step 2: Introduce a parsing layer separate from runtime sampling

Tasks:

- keep file reading and JSON translation separate from sampling logic
- create helpers that parse one record into a `ReceiptSample`
- normalize text and field values at the boundary, not ad hoc in the environment
- fail early on missing labels, malformed bounding boxes, or unusable OCR payloads

Suggested code changes:

- expand `env/dataset.py`
- add helper functions or a small parser module if needed

Acceptance criteria:

- one function is responsible for reading files
- one function is responsible for building domain models
- errors name the failing file or record ID

### Step 3: Preserve deterministic sampling

Tasks:

- maintain the current seeded selection contract
- allow filtering by difficulty, split, or task tag if needed
- ensure the same seed and task pick the same sample ordering on repeated runs

Acceptance criteria:

- determinism tests pass across repeated runs
- sampling remains stable even after replacing the mock records

### Step 4: Define dataset-to-difficulty mapping

Tasks:

- decide whether difficulty is driven by explicit split files, per-sample metadata, or runtime perturbation only
- avoid using sample ID substring matching as the long-term mechanism
- if the raw dataset has no difficulty labels, create a derived mapping layer that is explicit and testable

Acceptance criteria:

- difficulty selection logic is visible in code and does not depend on mock naming conventions

### Step 5: Add fixture-friendly tests

Tasks:

- create small fixture samples or use a tiny subset of the prepared dataset for tests
- validate that loading produces expected `ReceiptSample` fields
- validate deterministic sampling under a fixed seed
- validate failure modes for malformed input

Acceptance criteria:

- tests do not depend on the full dataset volume
- tests cover both success and failure paths

### Step 6: Retire or isolate the embedded mock records

Tasks:

- remove embedded samples from the default runtime path
- if they are still useful, move them into test fixtures or a separate fallback loader used only in development
- update any docs that still describe the mock dataset as the main behavior

Acceptance criteria:

- production runtime path no longer depends on `mock://` sample references

## File-Level Work Estimate

### Likely files to change

- `env/dataset.py`
- `env/models.py` if extra metadata is needed
- `tests/test_environment.py`
- a new dataset-focused test file
- possibly `scripts/prepare_sroie.py` if normalization needs to be formalized

## Testing Strategy

### Unit tests

- parse one valid annotation file into `ReceiptSample`
- reject malformed annotation payloads
- confirm deterministic sample selection under fixed seeds

### Integration tests

- `ReceiptExtractionEnv.reset()` succeeds on real dataset input
- environment can step through at least one episode without mock-only assumptions

### Manual verification

- print or inspect one loaded sample and confirm the gold fields align with annotation content

## Risks And Mitigations

### Risk: dataset format ambiguity

Mitigation:

- add parsing diagnostics early
- validate a handful of records before broadening the loader

### Risk: test brittleness due to external files

Mitigation:

- use stable mini fixtures for unit tests
- reserve full dataset runs for smoke or integration checks

### Risk: hidden normalization mismatch

Mitigation:

- normalize at load time where appropriate
- document exactly which fields are normalized before grading versus during grading

## Exit Criteria

This plan is complete when the default environment path uses prepared dataset records, deterministic sampling still works, and no submission-critical flow depends on the old embedded mock receipts.