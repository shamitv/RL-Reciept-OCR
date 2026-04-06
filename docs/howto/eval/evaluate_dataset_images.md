# Evaluating Dataset Images

The `scripts/evaluate_dataset_images.py` script is the primary tool for running bulk evaluations of LLM receipt extraction performance across your dataset.

## How It Works

The evaluation pipeline performs the following steps for each receipt image in the dataset:

1. **Extraction**: Calls your target LLM (defined by `MODEL_NAME` and `API_BASE_URL` in your `.env`) to extract receipt information such as the company name, date, address, and total amount.
2. **Deterministic Grading**: Automatically compares the LLM's predicted extraction against the gold annotations, assigning a deterministic score (0.0 to 1.0) for each field.
3. **LLM Judge**: Passes the extraction failures or discrepancies to a secondary Judge Model (e.g., `EVAL_MODEL` and `EVAL_API_BASE_URL`) to generate a textual evaluation, summary, and pinpoint specific failure reasons (e.g., Hallucination vs. Unclear text).
4. **Resiliency & Caching**: 
    - The LLM calls are routed through a local file-backed deterministic cache (`env.llm_cache`). If you run the exact same prompt configuration twice, you will not be charged twice for LLM requests!
    - The script safely skips over receipts it has already processed, avoiding redundant computation if the script gets interrupted.

### Storing Outputs

All processed results are stored by default in `artifacts/eval/dataset-image-eval`. 
You can expect the following three files to be managed by the script:

- `results.jsonl`: The core data-store. An append-only JSON Lines file storing the full `ReceiptEvalRecord` per receipt (predicted text, gold text, judge response, and individual field scores).
- `summary.json`: An aggregated summary showing the final mean score, total error counts, and top LLM failure reasons.
- `report.md`: A readable markdown file outlining the score of every receipt organized by Worked, Partial, Skipped, and Failed buckets.

## How To Invoke

To run an evaluation over your dataset, simply execute:

```bash
python scripts/evaluate_dataset_images.py
```

If the script fails halfway through, you can safely run the exact same command to resume where it left off.

If you change your prompts or want to perform a fresh run on already processed images, pass the `--force` flag:

```bash
python scripts/evaluate_dataset_images.py --force
```

## Command Line Options

| Flag | Description |
|---|---|
| `--force` | Force re-processing of images from scratch, ignoring previously cached runs in `results.jsonl`. |
| `--dataset-root <path>` | Target a specific dataset root folder. By default, it uses the `RECEIPT_DATASET_ROOT` environment variable or dynamically detects the bundled dataset inside the project. |
| `--output-dir <path>` | The directory where `results.jsonl`, `summary.json`, and `report.md` will be output. Defaults to `artifacts/eval/dataset-image-eval`. |
| `--limit <N>` | Run the script for a maximum of `N` images instead of the full dataset. Useful for debugging and smoke testing a small subset. |
| `-h`, `--help` | Show the help message and exit. |
