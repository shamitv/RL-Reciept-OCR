from fastapi.testclient import TestClient

from env.server import app


def test_server_endpoints_smoke() -> None:
    client = TestClient(app)

    root_response = client.get("/", headers={"accept": "application/json"})
    assert root_response.status_code == 200
    assert root_response.json()["status"] == "ok"
    assert root_response.json()["ui"] == "/eval"

    browser_root_response = client.get("/", headers={"accept": "text/html"}, follow_redirects=False)
    assert browser_root_response.status_code == 307
    assert browser_root_response.headers["location"] == "/eval"

    health_response = client.get("/healthz")
    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}

    ready_response = client.get("/readyz")
    assert ready_response.status_code == 200
    assert ready_response.json()["status"] == "ready"

    reset_response = client.post("/reset", params={"task_name": "easy", "seed": 7})
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert reset_payload["done"] is False

    step_response = client.post("/step", json={"action_type": "view_receipt"})
    assert step_response.status_code == 200
    assert "Viewing receipt" in step_response.json()["observation"]["last_action_result"]

    state_response = client.get("/state")
    assert state_response.status_code == 200
    assert state_response.json()["sample_id"]
