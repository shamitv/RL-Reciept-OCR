from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.config import load_environment
from env.evaluation import DatasetAuditRecord, audit_dataset, build_model_client, require_env, run_extraction_model
from env.graders import grade_receipt
from env.models import ReceiptDraft


DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "datasets" / "receipt-selection-50"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=json_default))
        handle.write("\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def score_bin(score: float) -> str:
    if score >= 0.9:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"


def line_item_bucket(count: int) -> str:
    if count == 0:
        return "none"
    if count == 1:
        return "one"
    if count <= 3:
        return "two_to_three"
    return "four_plus"


def process_record(record: DatasetAuditRecord, client: Any, model_name: str) -> dict[str, Any]:
    processed: dict[str, Any] = {
        "sample_id": record.sample_id,
        "task_id": record.task_id,
        "annotation_path": record.annotation_path,
        "image_path": record.image_path,
        "dataset_status": record.dataset_status,
        "skip_reason": record.skip_reason,
        "missing_categories": record.missing_categories,
        "unparseable_fields": record.unparseable_fields,
        "gold_fields": record.gold_fields.model_dump(mode="json") if record.gold_fields else None,
        "gold_line_items": [item.model_dump(mode="json") for item in record.gold_line_items],
        "predicted_fields": None,
        "predicted_line_items": [],
        "status": "skipped" if record.dataset_status != "runnable" else "failed",
        "score": 0.0,
        "success": False,
        "field_scores": {},
        "header_score": 0.0,
        "summary_score": 0.0,
        "line_items_score": 0.0,
        "reconciliation_score": 0.0,
        "reconciliation_status": None,
        "reconciliation_delta": None,
        "summary_reconciliation_status": None,
        "summary_reconciliation_delta": None,
        "line_item_reconciliation_status": None,
        "line_item_reconciliation_delta": None,
        "gold_line_item_count": len(record.gold_line_items),
        "predicted_line_item_count": 0,
        "line_item_count_score": None,
        "line_item_count_delta": None,
        "total_score": 0.0,
        "score_bin": "low",
        "gold_line_item_bucket": line_item_bucket(len(record.gold_line_items)),
        "error": None,
        "created_at": now_utc_iso(),
    }

    if record.dataset_status != "runnable":
        return processed

    try:
        prediction = run_extraction_model(record, client, model_name)
        grade = grade_receipt(
            prediction,
            record.gold_fields or ReceiptDraft(),
            task_id=record.task_id or "easy",
            gold_line_items=record.gold_line_items,
        )
        processed.update(
            {
                "predicted_fields": prediction.model_dump(mode="json"),
                "predicted_line_items": [item.model_dump(mode="json") for item in prediction.line_items],
                "status": "worked" if grade.success else ("partial" if grade.score > 0 else "failed"),
                "score": round(float(grade.score), 6),
                "success": bool(grade.success),
                "field_scores": {key: round(float(value), 6) for key, value in grade.field_scores.items()},
                "header_score": round(float(grade.header_score), 6),
                "summary_score": round(float(grade.summary_score), 6),
                "line_items_score": round(float(grade.line_items_score), 6),
                "reconciliation_score": round(float(grade.reconciliation_score), 6),
                "reconciliation_status": grade.reconciliation_status,
                "reconciliation_delta": grade.reconciliation_delta,
                "summary_reconciliation_status": grade.summary_reconciliation_status,
                "summary_reconciliation_delta": grade.summary_reconciliation_delta,
                "line_item_reconciliation_status": grade.line_item_reconciliation_status,
                "line_item_reconciliation_delta": grade.line_item_reconciliation_delta,
                "predicted_line_item_count": int(grade.predicted_line_item_count),
                "line_item_count_score": None if grade.line_item_count_score is None else round(float(grade.line_item_count_score), 6),
                "line_item_count_delta": grade.line_item_count_delta,
                "total_score": round(float(grade.field_scores.get("total", 0.0)), 6),
                "score_bin": score_bin(float(grade.score)),
            }
        )
    except Exception as exc:  # pragma: no cover - live model failures vary
        processed["error"] = str(exc)
    return processed


def processed_summary(records: list[dict[str, Any]], elapsed_seconds: float, model_name: str) -> dict[str, Any]:
    runnable = [record for record in records if record["dataset_status"] == "runnable"]
    scored = [record for record in runnable if not record.get("error")]
    return {
        "generated_at": now_utc_iso(),
        "model": model_name,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "elapsed_minutes": round(elapsed_seconds / 60.0, 3),
        "record_count": len(records),
        "runnable_count": len(runnable),
        "scored_count": len(scored),
        "status_counts": dict(Counter(record["status"] for record in records)),
        "dataset_status_counts": dict(Counter(record["dataset_status"] for record in records)),
        "task_counts": dict(Counter(record.get("task_id") or "none" for record in scored)),
        "score_bin_counts": dict(Counter(record["score_bin"] for record in scored)),
        "line_item_bucket_counts": dict(Counter(record["gold_line_item_bucket"] for record in scored)),
        "reconciliation_status_counts": dict(Counter(record.get("reconciliation_status") or "not_evaluated" for record in scored)),
        "mean_score": round(mean(record["score"] for record in scored), 6) if scored else 0.0,
        "min_score": round(min((record["score"] for record in scored), default=0.0), 6),
        "max_score": round(max((record["score"] for record in scored), default=0.0), 6),
        "records_with_errors": sum(1 for record in records if record.get("error")),
    }


def pick_from_order(
    selected: dict[str, dict[str, Any]],
    reasons: dict[str, list[str]],
    ordered_records: list[dict[str, Any]],
    reason: str,
    count: int,
    target_size: int,
) -> None:
    added = 0
    for record in ordered_records:
        sample_id = record["sample_id"]
        if sample_id in selected:
            if reason not in reasons[sample_id]:
                reasons[sample_id].append(reason)
            continue
        selected[sample_id] = record
        reasons[sample_id] = [reason]
        added += 1
        if len(selected) >= target_size or added >= count:
            return


def select_records(records: list[dict[str, Any]], target_size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = [
        record
        for record in records
        if record["dataset_status"] == "runnable" and record.get("image_path") and record.get("annotation_path") and not record.get("error")
    ]
    if len(candidates) < target_size:
        raise ValueError(f"Only {len(candidates)} scored runnable records are available; cannot select {target_size}")

    selected: dict[str, dict[str, Any]] = {}
    reasons: dict[str, list[str]] = {}
    by_low_score = sorted(candidates, key=lambda item: (item["score"], item["sample_id"]))
    by_high_score = sorted(candidates, key=lambda item: (-item["score"], item["sample_id"]))
    median_score = sorted(record["score"] for record in candidates)[len(candidates) // 2]
    by_mid_score = sorted(candidates, key=lambda item: (abs(item["score"] - median_score), item["sample_id"]))

    pick_from_order(selected, reasons, by_low_score, "lowest_overall_scores", 10, target_size)
    pick_from_order(selected, reasons, by_high_score, "highest_overall_scores", 10, target_size)
    pick_from_order(selected, reasons, by_mid_score, "middle_overall_scores", 8, target_size)

    total_issues = [record for record in by_low_score if record.get("total_score", 0.0) < 1.0]
    pick_from_order(selected, reasons, total_issues, "total_field_errors", 6, target_size)

    rich_line_items = sorted(candidates, key=lambda item: (-item.get("gold_line_item_count", 0), item["sample_id"]))
    pick_from_order(selected, reasons, rich_line_items, "line_item_count_spectrum", 6, target_size)

    line_item_count_issues = sorted(
        [record for record in candidates if record.get("line_item_count_delta") not in (None, 0)],
        key=lambda item: (-(item.get("line_item_count_delta") or 0), item["sample_id"]),
    )
    pick_from_order(selected, reasons, line_item_count_issues, "line_item_count_mismatch", 6, target_size)

    line_item_reconciliation_issues = [
        record for record in by_low_score if record.get("line_item_reconciliation_status") in {"fail", "partial"}
    ]
    pick_from_order(selected, reasons, line_item_reconciliation_issues, "line_item_reconciliation_issues", 5, target_size)

    summary_reconciliation_issues = [
        record for record in by_low_score if record.get("summary_reconciliation_status") in {"fail", "partial"}
    ]
    pick_from_order(selected, reasons, summary_reconciliation_issues, "summary_reconciliation_issues", 5, target_size)

    for task_id in ("easy", "medium", "hard"):
        task_records = sorted(
            [record for record in candidates if record.get("task_id") == task_id],
            key=lambda item: (item["score"], -item.get("gold_line_item_count", 0), item["sample_id"]),
        )
        pick_from_order(selected, reasons, task_records, f"task_balance_{task_id}", 6, target_size)

    diversity_order = sorted(
        candidates,
        key=lambda item: (
            item["score_bin"],
            item.get("task_id") or "",
            item["gold_line_item_bucket"],
            item.get("reconciliation_status") or "",
            item["sample_id"],
        ),
    )
    pick_from_order(selected, reasons, diversity_order, "diversity_fill", target_size, target_size)

    selected_records = list(selected.values())[:target_size]
    for record in selected_records:
        record["selection_reasons"] = reasons[record["sample_id"]]

    summary = {
        "selected_count": len(selected_records),
        "target_size": target_size,
        "score_range": {
            "min": round(min(record["score"] for record in selected_records), 6),
            "max": round(max(record["score"] for record in selected_records), 6),
            "mean": round(mean(record["score"] for record in selected_records), 6),
        },
        "task_counts": dict(Counter(record.get("task_id") or "none" for record in selected_records)),
        "score_bin_counts": dict(Counter(record["score_bin"] for record in selected_records)),
        "total_score_counts": dict(Counter("correct" if record.get("total_score") == 1.0 else "not_correct" for record in selected_records)),
        "gold_line_item_bucket_counts": dict(Counter(record["gold_line_item_bucket"] for record in selected_records)),
        "reconciliation_status_counts": dict(Counter(record.get("reconciliation_status") or "not_evaluated" for record in selected_records)),
        "summary_reconciliation_status_counts": dict(
            Counter(record.get("summary_reconciliation_status") or "not_evaluated" for record in selected_records)
        ),
        "line_item_reconciliation_status_counts": dict(
            Counter(record.get("line_item_reconciliation_status") or "not_evaluated" for record in selected_records)
        ),
        "selection_reason_counts": dict(Counter(reason for record in selected_records for reason in record["selection_reasons"])),
    }
    return selected_records, summary


def copy_subset_files(selected_records: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    dataset_dir = output_dir / "dataset"
    ann_dir = dataset_dir / "ann"
    img_dir = dataset_dir / "img"
    ann_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    for record in selected_records:
        annotation_path = Path(record["annotation_path"])
        image_path = Path(record["image_path"])
        if annotation_path.exists():
            shutil.copy2(annotation_path, ann_dir / annotation_path.name)
        if image_path.exists():
            shutil.copy2(image_path, img_dir / image_path.name)
    return {"dataset_dir": str(dataset_dir), "ann_dir": str(ann_dir), "img_dir": str(img_dir)}


def write_selected_csv(selected_records: list[dict[str, Any]], output_dir: Path) -> None:
    csv_path = output_dir / "selected_samples.csv"
    fieldnames = [
        "sample_id",
        "task_id",
        "score",
        "total_score",
        "gold_line_item_count",
        "predicted_line_item_count",
        "line_item_count_delta",
        "line_items_score",
        "reconciliation_status",
        "summary_reconciliation_status",
        "line_item_reconciliation_status",
        "selection_reasons",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in selected_records:
            writer.writerow(
                {
                    key: ",".join(record[key]) if key == "selection_reasons" else record.get(key)
                    for key in fieldnames
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process receipts once with the LLM and build a score-diverse 50-image subset.")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--subset-size", type=int, default=50)
    parser.add_argument("--force", action="store_true", help="Reprocess records even if processed_results.jsonl exists.")
    return parser


def main() -> int:
    load_environment()
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    processed_path = output_dir / "processed_results.jsonl"
    summary_path = output_dir / "processed_summary.json"

    start = time.perf_counter()
    model_name = require_env("MODEL_NAME")
    require_env("API_BASE_URL")
    require_env("HF_TOKEN")

    if args.force and processed_path.exists():
        processed_path.unlink()

    records = load_jsonl(processed_path)
    processed_sample_ids = {record["sample_id"] for record in records}
    audits = audit_dataset(args.dataset_root)
    pending = [record for record in audits if record.sample_id not in processed_sample_ids]

    if pending:
        client = build_model_client(require_env("API_BASE_URL"))
        for index, record in enumerate(pending, start=1):
            processed = process_record(record, client, model_name)
            append_jsonl(processed_path, processed)
            records.append(processed)
            print(
                f"[PROCESS] {index}/{len(pending)} sample={record.sample_id} "
                f"status={processed['status']} score={processed['score']:.3f}",
                flush=True,
            )

    elapsed_seconds = time.perf_counter() - start
    summary = processed_summary(records, elapsed_seconds, model_name)
    write_json(summary_path, summary)

    selected_records, selection_summary = select_records(records, args.subset_size)
    copied_paths = copy_subset_files(selected_records, output_dir)
    write_json(output_dir / "selected_manifest.json", {"summary": selection_summary, "records": selected_records, "paths": copied_paths})
    selected_jsonl = output_dir / "selected_records.jsonl"
    selected_jsonl.write_text("", encoding="utf-8")
    for record in selected_records:
        append_jsonl(selected_jsonl, record)
    write_selected_csv(selected_records, output_dir)

    write_json(
        output_dir / "README.json",
        {
            "purpose": "LLM-processed receipt subset selected for score and receipt-structure diversity.",
            "processed_results": str(processed_path),
            "processed_summary": str(summary_path),
            "selected_manifest": str(output_dir / "selected_manifest.json"),
            "selected_records": str(selected_jsonl),
            "selected_csv": str(output_dir / "selected_samples.csv"),
            "copied_dataset": copied_paths,
        },
    )

    print(f"[SUMMARY] processed={len(records)} selected={len(selected_records)} output_dir={output_dir}", flush=True)
    print(f"[SUMMARY] elapsed_seconds={elapsed_seconds:.3f} elapsed_minutes={elapsed_seconds / 60.0:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
