import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_openenv_cli() -> str:
    script_name = "openenv.exe" if os.name == "nt" else "openenv"
    sibling_path = Path(sys.executable).with_name(script_name)
    if sibling_path.exists():
        return str(sibling_path)

    executable = shutil.which("openenv")
    if executable:
        return executable

    raise AssertionError("openenv CLI is not installed in the active Python environment")


def test_openenv_validate_passes() -> None:
    result = subprocess.run(
        [_find_openenv_cli(), "validate"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    combined_output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    assert result.returncode == 0, combined_output