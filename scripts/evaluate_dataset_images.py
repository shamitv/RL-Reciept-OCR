from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.config import load_environment
from env.evaluation import evaluate_dataset_images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run receipt image evaluation across the full dataset.")
    parser.add_argument("--dataset-root", default=None, help="Override the dataset root. Defaults to RECEIPT_DATASET_ROOT or the bundled dataset.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for results.jsonl, summary.json, and report.md. Defaults to artifacts/eval/dataset-image-eval.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N dataset records after resume filtering.")
    parser.add_argument("--resume", action="store_true", help="Skip sample IDs that already exist in results.jsonl.")
    return parser


def main() -> int:
    load_environment()
    parser = build_parser()
    args = parser.parse_args()
    result = evaluate_dataset_images(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        limit=args.limit,
        resume=args.resume,
    )
    print(
        (
            "[EVAL] processed_records={processed} expected_total_records={expected} "
            "results={results} summary={summary} report={report}"
        ).format(
            processed=result.processed_records,
            expected=result.expected_total_records,
            results=result.results_path,
            summary=result.summary_path,
            report=result.report_path,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
