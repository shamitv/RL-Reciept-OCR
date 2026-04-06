from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_environment(env_file: str | Path | None = None) -> None:
    if env_file is None:
        env_path = Path(__file__).resolve().parents[1] / ".env"
    else:
        env_path = Path(env_file)
    load_dotenv(env_path, override=False)