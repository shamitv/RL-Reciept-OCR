from pathlib import Path

from env.config import load_environment


def test_load_environment_reads_dotenv_file(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_NAME=test-model\nAPI_BASE_URL=https://example.test/v1\n", encoding="utf-8")

    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)

    load_environment(env_file)

    assert env_file.exists()
    assert Path(env_file).read_text(encoding="utf-8")
    assert __import__("os").getenv("MODEL_NAME") == "test-model"
    assert __import__("os").getenv("API_BASE_URL") == "https://example.test/v1"


def test_load_environment_does_not_override_existing_values(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_NAME=file-model\n", encoding="utf-8")

    monkeypatch.setenv("MODEL_NAME", "existing-model")

    load_environment(env_file)

    assert __import__("os").getenv("MODEL_NAME") == "existing-model"