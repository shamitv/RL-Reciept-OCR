from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.environment import ReceiptExtractionEnv
from env.models import ReceiptAction


def main() -> None:
    env = ReceiptExtractionEnv()
    result = env.reset(task_name="easy", seed=7)
    print(result.observation.model_dump_json(indent=2))
    result = env.step(ReceiptAction(action_type="view_receipt"))
    print(result.observation.last_action_result)


if __name__ == "__main__":
    main()
