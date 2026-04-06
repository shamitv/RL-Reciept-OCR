from pathlib import Path
import sys

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.environment import ReceiptExtractionEnv
from env.models import ReceiptAction
from env.server import app
from inference import evaluate_tasks


def main() -> None:
    env = ReceiptExtractionEnv()
    result = env.reset(task_name="easy", seed=7)
    assert result.done is False
    print(result.observation.model_dump_json(indent=2))

    result = env.step(ReceiptAction(action_type="view_receipt"))
    assert "Viewing receipt" in result.observation.last_action_result
    print(result.observation.last_action_result)

    client = TestClient(app)
    reset_response = client.post("/reset", params={"task_name": "easy", "seed": 7})
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert reset_payload["done"] is False

    step_response = client.post("/step", json={"action_type": "view_receipt"})
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert "Viewing receipt" in step_payload["observation"]["last_action_result"]

    state_response = client.get("/state")
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert state_payload["sample_id"]

    summary = evaluate_tasks(tasks=["easy", "medium", "hard"], seed=7, episodes=1, verbose=False)
    assert summary["aggregate"]["task_count"] == 3
    print(summary)


if __name__ == "__main__":
    main()
