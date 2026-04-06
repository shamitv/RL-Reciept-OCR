# OpenEnv Receipt Understanding Baseline Scores

Generated on `2026-04-06` from the current repository state.

## Command

```powershell
.\.venv\Scripts\python.exe inference.py --format json
```

## Configuration

- Agent: `heuristic`
- Base seed: `7`
- Episodes per task: `1`

## Aggregate

- Task count: `3`
- Mean score: `0.233333`
- Mean success rate: `0.0`
- Mean steps: `10`
- Mean cumulative reward: `0.26`

## Task Results

### Easy

- Mean score: `0.400`
- Success rate: `0.0`
- Mean steps: `12`
- Mean cumulative reward: `0.44`
- Sample: `1091-receipt.jpg`
- Field scores:
  - `company`: `0.0`
  - `date`: `1.0`
  - `address`: `0.8`
  - `subtotal`: `0.0`
  - `tax`: `0.0`
  - `total`: `0.0`

### Medium

- Mean score: `0.200`
- Success rate: `0.0`
- Mean steps: `10`
- Mean cumulative reward: `0.23`
- Sample: `1017-receipt.jpg`
- Field scores:
  - `company`: `1.0`
  - `date`: `1.0`
  - `address`: `0.0`
  - `subtotal`: `0.0`
  - `tax`: `0.0`
  - `total`: `0.0`

### Hard

- Mean score: `0.100`
- Success rate: `0.0`
- Mean steps: `8`
- Mean cumulative reward: `0.11`
- Sample: `1065-receipt.jpg`
- Field scores:
  - `company`: `1.0`
  - `date`: `1.0`
  - `address`: `0.0`
  - `subtotal`: `0.0`
  - `tax`: `0.0`
  - `total`: `0.0`

## Notes

- These numbers reflect the current receipt-native task definitions:
  - `easy`: header extraction
  - `medium`: summary extraction plus reconciliation
  - `hard`: summary extraction plus line items and reconciliation
- Re-run the command above after any change to tasks, grading, rewards, agents, or dataset parsing.
