from env.candidate_retrieval import query_candidates
from env.environment import ReceiptExtractionEnv
from env.models import OCRRegion, ReceiptAction, ReceiptDraft, ReceiptSample


def make_sample() -> ReceiptSample:
    return ReceiptSample(
        sample_id="synthetic-receipt",
        image_ref="synthetic://receipt",
        regions=[
            OCRRegion(region_id="top-company", text="Demo Shop", bbox=(0, 10, 100, 30)),
            OCRRegion(region_id="middle-address", text="123 Main Street", bbox=(0, 120, 140, 140)),
            OCRRegion(region_id="middle-date", text="25/03/2019", bbox=(0, 145, 100, 165)),
            OCRRegion(region_id="bottom-total-a", text="TOTAL 18.50", bbox=(0, 220, 120, 240)),
            OCRRegion(region_id="bottom-total-b", text="AMOUNT DUE 18.40", bbox=(0, 235, 135, 255)),
        ],
        gold_fields=ReceiptDraft(company="Demo Shop", address="123 Main Street", date="2019-03-25", total="18.50"),
    )


def reset_with_sample(task_name: str) -> ReceiptExtractionEnv:
    env = ReceiptExtractionEnv()
    sample = make_sample()
    env.dataset.sample = lambda difficulty, rng: sample.model_copy(deep=True)
    env.reset(task_name=task_name, seed=3)
    return env


def test_view_receipt_reveals_more_regions_on_easy_than_medium() -> None:
    easy_env = reset_with_sample("easy")
    medium_env = reset_with_sample("medium")

    easy_result = easy_env.step(ReceiptAction(action_type="view_receipt"))
    medium_result = medium_env.step(ReceiptAction(action_type="view_receipt"))

    assert len(easy_result.observation.visible_regions) > len(medium_result.observation.visible_regions)


def test_task_rejects_unavailable_windows() -> None:
    medium_env = reset_with_sample("medium")
    hard_env = reset_with_sample("hard")

    medium_result = medium_env.step(ReceiptAction(action_type="list_text_regions", window="all"))
    hard_result = hard_env.step(ReceiptAction(action_type="list_text_regions", window="middle"))

    assert medium_result.observation.last_action_result == "Window all is unavailable for medium"
    assert hard_result.observation.last_action_result == "Window middle is unavailable for hard"


def test_candidate_reranking_is_deterministic_and_task_sensitive() -> None:
    regions = make_sample().regions

    easy_candidates = query_candidates("total", regions, ranking_noise=0.0, noise_key="sample:easy:total")
    hard_candidates_first = query_candidates("total", regions, ranking_noise=0.375, noise_key="sample:hard:total")
    hard_candidates_second = query_candidates("total", regions, ranking_noise=0.375, noise_key="sample:hard:total")

    assert hard_candidates_first == hard_candidates_second
    assert [candidate.heuristic_score for candidate in easy_candidates] != [candidate.heuristic_score for candidate in hard_candidates_first]