from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(label: str, command: list[str]) -> None:
    print(f"[VALIDATE] {label}: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    python_exe = sys.executable
    run_step("pytest", [python_exe, "-m", "pytest"])
    run_step("smoke_test", [python_exe, "scripts/smoke_test.py"])
    run_step("baseline", [python_exe, "inference.py", "--format", "text"])


if __name__ == "__main__":
    main()