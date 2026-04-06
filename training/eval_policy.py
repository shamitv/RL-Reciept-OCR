from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference import main as inference_main


def main() -> int:
    argv = list(sys.argv[1:])
    if "--agent" not in argv:
        argv = ["--agent", "ppo", *argv]
    return inference_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
