from env.environment import ReceiptExtractionEnv
from env.models import ReceiptAction


def main() -> None:
    env = ReceiptExtractionEnv()
    env.reset(task_name="easy", seed=0)
    scripted = [
        ReceiptAction(action_type="view_receipt"),
        ReceiptAction(action_type="list_text_regions", window="all"),
        ReceiptAction(action_type="query_candidates", field="company"),
    ]
    for action in scripted:
        env.step(action)
    print("behavior cloning data builder placeholder")


if __name__ == "__main__":
    main()
