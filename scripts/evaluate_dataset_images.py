from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


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
    parser.add_argument("--force", action="store_true", help="Force re-processing of images even if they already exist in results.jsonl.")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("Starting dataset evaluation script")
    
    load_environment()
    parser = build_parser()
    args = parser.parse_args()
    
    resume = not args.force
    logger.info(f"Running evaluation (limit={args.limit}, resume={resume})")
    
    result = evaluate_dataset_images(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        limit=args.limit,
        resume=resume,
    )
    
    logger.info(
        "Evaluation completed.\n"
        f"  Processed records: {result.processed_records}\n"
        f"  Expected total: {result.expected_total_records}\n"
        f"  Results file: {result.results_path}\n"
        f"  Summary file: {result.summary_path}\n"
        f"  Report file: {result.report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
