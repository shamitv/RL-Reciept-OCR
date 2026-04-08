from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.heuristic import HeuristicAgent
from env.environment import ReceiptExtractionEnv
from env.models import ReceiptAction, ReceiptDraft, ReceiptLineItem
from inference import (
    TASK_ORDER,
    actions_from_llm_prediction,
    build_agent,
    episode_seed,
    evaluate_tasks,
    load_selected_audit_records,
    resolve_tasks,
    run_episode,
    run_llm_episode,
)


@dataclass(frozen=True)
class WrappedHeuristicAgent:
    name: str = "ppo-mock"

    def select_action(self, env: ReceiptExtractionEnv) -> ReceiptAction:
        return HeuristicAgent().select_action(env)


class _FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        content = self.responses[index]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.completions = _FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)
        self.base_url = "https://fake.client/v1"


def _write_image_json(root: Path, image_id: str, payload: bytes = b"image") -> Path:
    image_json_dir = root / "img_json"
    image_json_dir.mkdir(parents=True, exist_ok=True)
    path = image_json_dir / f"{image_id}.json"
    path.write_text(
        json.dumps(
            {
                "image_id": image_id,
                "mime_type": "image/jpeg",
                "image_data": base64.b64encode(payload).decode("ascii"),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_resolve_tasks_defaults_to_all_tasks() -> None:
    assert resolve_tasks("all") == list(TASK_ORDER)


def test_episode_seed_is_deterministic_and_task_specific() -> None:
    assert episode_seed(7, "easy", 0) == 7
    assert episode_seed(7, "medium", 0) == 1007
    assert episode_seed(7, "hard", 2) == 2009


def test_run_episode_is_deterministic_for_fixed_seed() -> None:
    first = run_episode(task="easy", seed=7)
    second = run_episode(task="easy", seed=7)

    assert first == second


def test_evaluate_tasks_returns_all_task_summary() -> None:
    summary = evaluate_tasks(tasks=list(TASK_ORDER), seed=7, episodes=1)

    assert summary["agent"] == "heuristic"
    assert summary["episodes_per_task"] == 1
    assert [task_summary["task"] for task_summary in summary["tasks"]] == list(TASK_ORDER)
    assert summary["aggregate"]["task_count"] == 3


def test_build_agent_defaults_to_heuristic() -> None:
    agent = build_agent()

    assert isinstance(agent, HeuristicAgent)


def test_build_agent_requires_checkpoint_for_ppo() -> None:
    with pytest.raises(ValueError, match="--checkpoint is required when --agent ppo"):
        build_agent(agent_name="ppo")


def test_custom_agent_name_flows_into_summary() -> None:
    summary = evaluate_tasks(tasks=["easy"], seed=7, episodes=1, agent=WrappedHeuristicAgent())

    assert summary["agent"] == "ppo-mock"


def test_custom_agent_keeps_summary_schema() -> None:
    heuristic_summary = evaluate_tasks(tasks=["easy"], seed=7, episodes=1)
    wrapped_summary = evaluate_tasks(tasks=["easy"], seed=7, episodes=1, agent=WrappedHeuristicAgent())

    assert set(heuristic_summary) == set(wrapped_summary)
    assert set(heuristic_summary["aggregate"]) == set(wrapped_summary["aggregate"])


def test_actions_from_llm_prediction_include_manual_line_items() -> None:
    prediction = ReceiptDraft(
        company="City Cafe",
        date="2019-03-27",
        subtotal="8.00",
        tax="0.48",
        total="8.48",
        line_items=[ReceiptLineItem(description="Latte", line_total="4.50")],
    )

    actions = actions_from_llm_prediction(
        prediction=prediction,
        field_order=["company", "date", "subtotal", "tax", "total"],
        requires_line_items=True,
        max_actions=10,
    )

    assert actions[-1].action_type == "add_line_item_manual"
    assert actions[-1].value == "Latte"
    assert actions[-1].line_total == "4.50"


def test_run_llm_episode_uses_existing_extraction_client(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path / "llm-cache"))
    client = _FakeClient(
        [
            (
                '{"company":"Example Store","date":"2019-03-25",'
                '"address":"123 Main St","subtotal":null,"tax":null,"total":"31.00","line_items":[]}'
            )
        ]
    )

    result = run_llm_episode(task="easy", seed=7, client=client, model_name="llm-model", emit_logs=True)
    output_lines = capsys.readouterr().out.strip().splitlines()

    assert client.completions.calls
    assert output_lines[0] == "[START] task=easy env=rl-receipt-ocr model=llm-model"
    assert all(line.startswith(("[START]", "[STEP]", "[END]")) for line in output_lines)
    assert output_lines[-1].startswith("[END] success=")
    assert result["steps"] >= 1


def test_load_selected_audit_records_prefers_copied_dataset_paths(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    image_json_path = dataset_dir / "img_json" / "sample-receipt.jpg.json"
    annotation_path = dataset_dir / "ann" / "sample-receipt.jpg.json"
    annotation_path.parent.mkdir(parents=True)
    image_json_path = _write_image_json(dataset_dir, "sample-receipt.jpg")
    annotation_path.write_text("{}", encoding="utf-8")

    manifest_path = tmp_path / "selected_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "sample_id": "sample-receipt.jpg",
                        "task_id": "easy",
                        "dataset_status": "runnable",
                        "image_id": "sample-receipt.jpg",
                        "image_json_path": "D:/stale/img_json/sample-receipt.jpg.json",
                        "annotation_path": "D:/stale/ann/sample-receipt.jpg.json",
                        "gold_fields": {
                            "company": "Store",
                            "date": "2019-03-25",
                            "address": "1 Road",
                            "total": "10.00",
                        },
                        "gold_line_items": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    records = load_selected_audit_records(manifest_path)

    assert len(records) == 1
    assert records[0].sample_id == "sample-receipt.jpg"
    assert records[0].image_id == "sample-receipt.jpg"
    assert records[0].image_json_path == str(image_json_path)
    assert records[0].annotation_path == str(annotation_path)


def test_load_selected_audit_records_falls_back_to_repo_dataset_image_json(tmp_path: Path) -> None:
    repo_dataset_dir = tmp_path / "dataset" / "Receipt dataset" / "ds0"
    repo_annotation_path = repo_dataset_dir / "ann" / "sample-receipt.jpg.json"
    repo_annotation_path.parent.mkdir(parents=True)
    repo_image_json_path = _write_image_json(repo_dataset_dir, "sample-receipt.jpg")
    repo_annotation_path.write_text("{}", encoding="utf-8")

    selected_dataset_dir = tmp_path / "artifacts" / "datasets" / "receipt-selection-50" / "dataset"
    selected_annotation_path = selected_dataset_dir / "ann" / "sample-receipt.jpg.json"
    selected_annotation_path.parent.mkdir(parents=True)
    selected_annotation_path.write_text("{}", encoding="utf-8")

    manifest_path = tmp_path / "artifacts" / "datasets" / "receipt-selection-50" / "selected_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "sample_id": "sample-receipt.jpg",
                        "task_id": "easy",
                        "dataset_status": "runnable",
                        "image_id": "sample-receipt.jpg",
                        "image_json_path": "D:/stale/img_json/sample-receipt.jpg.json",
                        "annotation_path": "D:/stale/ann/sample-receipt.jpg.json",
                        "gold_fields": {
                            "company": "Store",
                            "date": "2019-03-25",
                            "address": "1 Road",
                            "total": "10.00",
                        },
                        "gold_line_items": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    records = load_selected_audit_records(manifest_path)

    assert len(records) == 1
    assert records[0].image_json_path == str(repo_image_json_path)
    assert records[0].annotation_path == str(selected_annotation_path)
