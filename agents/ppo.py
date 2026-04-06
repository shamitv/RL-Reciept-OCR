from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.encoding import ENCODER_VERSION, FIELD_ORDER, OBSERVATION_DIM, encode_observation
from env.environment import ReceiptExtractionEnv
from env.models import ReceiptAction

SUPPORTED_PPO_ACTIONS = (
    "view_receipt",
    "list_text_regions",
    "inspect_bbox",
    "inspect_neighbors",
    "query_candidates",
    "set_field_from_candidate",
    "query_line_item_candidates",
    "add_line_item_from_candidate",
    "remove_line_item",
    "normalize_field",
    "check_total_consistency",
    "check_date_format",
    "check_receipt_consistency",
    "clear_field",
    "submit",
)
WINDOW_ORDER = ("all", "top", "middle", "bottom")
RADIUS_BUCKETS = (1, 2, 3)
PARAMETER_HEAD_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "list_text_regions": ("window",),
    "inspect_bbox": ("bbox_index",),
    "inspect_neighbors": ("bbox_index", "radius_bucket"),
    "query_candidates": ("field",),
    "set_field_from_candidate": ("field", "candidate_index"),
    "add_line_item_from_candidate": ("candidate_index",),
    "remove_line_item": ("line_item_index",),
    "normalize_field": ("field",),
    "clear_field": ("field",),
}
REQUIRED_CHECKPOINT_FIELDS = (
    "model_state_dict",
    "obs_dim",
    "action_types",
    "param_heads",
    "architecture",
    "encoder_version",
)


class CheckpointIncompatibleError(RuntimeError):
    """Raised when a PPO checkpoint does not match the runtime contract."""


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised by runtime error paths
        raise ImportError(
            "torch is required for --agent ppo. Install with: pip install -e '.[ppo]'"
        ) from exc
    return torch


def active_fields(env: ReceiptExtractionEnv) -> list[str]:
    return [field for field in FIELD_ORDER if field in env.task.target_fields]


def filled_fields(env: ReceiptExtractionEnv) -> list[str]:
    obs = env.last_observation
    return [field for field in active_fields(env) if getattr(obs.current_draft, field) is not None]


def queryable_fields(env: ReceiptExtractionEnv) -> list[str]:
    obs = env.last_observation
    return [
        field
        for field in active_fields(env)
        if not (obs.candidate_lists.get(field) and getattr(obs.current_draft, field) is not None)
    ]


def candidate_fields(env: ReceiptExtractionEnv) -> list[str]:
    obs = env.last_observation
    return [field for field in active_fields(env) if obs.candidate_lists.get(field)]


def visible_region_ids(env: ReceiptExtractionEnv) -> list[str]:
    return [region.region_id for region in env.last_observation.visible_regions]


def action_type_mask(env: ReceiptExtractionEnv) -> dict[str, bool]:
    obs = env.last_observation
    filled = filled_fields(env)
    visible_ids = visible_region_ids(env)
    mask = {action_type: False for action_type in SUPPORTED_PPO_ACTIONS}
    mask["view_receipt"] = obs.step_index == 0
    mask["list_text_regions"] = bool(env.task.visible_windows)
    mask["inspect_bbox"] = bool(visible_ids)
    mask["inspect_neighbors"] = bool(visible_ids)
    mask["query_candidates"] = bool(queryable_fields(env))
    mask["set_field_from_candidate"] = bool(candidate_fields(env))
    mask["query_line_item_candidates"] = bool(env.task.requires_line_items and visible_ids)
    mask["add_line_item_from_candidate"] = bool(env.task.requires_line_items and obs.line_item_candidates)
    mask["remove_line_item"] = bool(env.task.requires_line_items and obs.current_draft.line_items)
    mask["normalize_field"] = bool(filled)
    mask["check_total_consistency"] = obs.current_draft.total is not None
    mask["check_date_format"] = obs.current_draft.date is not None
    mask["check_receipt_consistency"] = env.task.task_id in {"medium", "hard"} and all(
        getattr(obs.current_draft, field) is not None for field in ("subtotal", "tax", "total")
    )
    mask["clear_field"] = bool(filled)
    mask["submit"] = bool(obs.terminal_allowed and (filled or obs.current_draft.line_items))
    return mask


def parameter_choices(env: ReceiptExtractionEnv, action_type: str) -> dict[str, list[Any]]:
    if action_type == "list_text_regions":
        return {"window": [window for window in WINDOW_ORDER if window in env.task.visible_windows]}
    if action_type == "inspect_bbox":
        return {"bbox_index": visible_region_ids(env)}
    if action_type == "inspect_neighbors":
        return {"bbox_index": visible_region_ids(env), "radius_bucket": list(RADIUS_BUCKETS)}
    if action_type == "query_candidates":
        return {"field": queryable_fields(env)}
    if action_type == "set_field_from_candidate":
        return {"field": candidate_fields(env)}
    if action_type == "add_line_item_from_candidate":
        return {"candidate_index": [candidate.candidate_id for candidate in env.last_observation.line_item_candidates]}
    if action_type == "remove_line_item":
        return {"line_item_index": list(range(len(env.last_observation.current_draft.line_items)))}
    if action_type == "normalize_field":
        return {"field": filled_fields(env)}
    if action_type == "clear_field":
        return {"field": filled_fields(env)}
    return {}


def _validate_param_heads(param_heads: dict[str, dict[str, int]]) -> None:
    for action_type, required_params in PARAMETER_HEAD_REQUIREMENTS.items():
        head_config = param_heads.get(action_type)
        if head_config is None:
            raise CheckpointIncompatibleError(f"checkpoint missing param_heads for {action_type}")
        missing = [name for name in required_params if name not in head_config]
        if missing:
            raise CheckpointIncompatibleError(
                f"checkpoint param_heads for {action_type} missing: {', '.join(missing)}"
            )
        for name, size in head_config.items():
            if not isinstance(size, int) or size <= 0:
                raise CheckpointIncompatibleError(f"invalid param_heads size for {action_type}.{name}")

    field_head_actions = ("query_candidates", "set_field_from_candidate", "normalize_field", "clear_field")
    for action_type in field_head_actions:
        if param_heads[action_type]["field"] != len(FIELD_ORDER):
            raise CheckpointIncompatibleError(f"{action_type}.field head must have {len(FIELD_ORDER)} classes")
    if param_heads["list_text_regions"]["window"] != len(WINDOW_ORDER):
        raise CheckpointIncompatibleError("list_text_regions.window head must have 4 classes")
    if param_heads["inspect_neighbors"]["radius_bucket"] != len(RADIUS_BUCKETS):
        raise CheckpointIncompatibleError("inspect_neighbors.radius_bucket head must have 3 classes")


def validate_checkpoint_payload(payload: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_CHECKPOINT_FIELDS if field not in payload]
    if missing:
        raise CheckpointIncompatibleError(f"checkpoint missing required fields: {', '.join(missing)}")

    if payload["obs_dim"] != OBSERVATION_DIM:
        raise CheckpointIncompatibleError(
            f"checkpoint obs_dim {payload['obs_dim']} does not match runtime encoder dim {OBSERVATION_DIM}"
        )
    if payload["encoder_version"] != ENCODER_VERSION:
        raise CheckpointIncompatibleError(
            f"checkpoint encoder_version {payload['encoder_version']} does not match runtime {ENCODER_VERSION}"
        )
    if list(payload["action_types"]) != list(SUPPORTED_PPO_ACTIONS):
        raise CheckpointIncompatibleError("checkpoint action_types do not match the supported PPO action subset")

    param_heads = payload["param_heads"]
    if not isinstance(param_heads, dict):
        raise CheckpointIncompatibleError("checkpoint param_heads must be a dict")
    _validate_param_heads(param_heads)

    architecture = payload["architecture"]
    if not isinstance(architecture, dict):
        raise CheckpointIncompatibleError("checkpoint architecture must be a dict")
    hidden_sizes = architecture.get("hidden_sizes")
    if not isinstance(hidden_sizes, list) or not hidden_sizes or not all(isinstance(size, int) and size > 0 for size in hidden_sizes):
        raise CheckpointIncompatibleError("checkpoint architecture.hidden_sizes must be a non-empty list of positive ints")
    if architecture.get("activation") not in {"relu", "tanh"}:
        raise CheckpointIncompatibleError("checkpoint architecture.activation must be 'relu' or 'tanh'")


def _activation_class(nn: Any, activation_name: str):
    if activation_name == "relu":
        return nn.ReLU
    if activation_name == "tanh":
        return nn.Tanh
    raise CheckpointIncompatibleError(f"unsupported activation {activation_name}")


def build_policy_network(
    torch_module: Any,
    obs_dim: int,
    action_types: tuple[str, ...],
    architecture: dict[str, Any],
    param_heads: dict[str, dict[str, int]],
):
    nn = torch_module.nn
    activation = _activation_class(nn, architecture["activation"])
    layers: list[Any] = []
    input_dim = obs_dim
    for hidden_size in architecture["hidden_sizes"]:
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(activation())
        input_dim = hidden_size
    feature_dim = input_dim
    trunk = nn.Sequential(*layers) if layers else nn.Identity()

    class PolicyNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = trunk
            self.action_head = nn.Linear(feature_dim, len(action_types))
            self.param_heads = nn.ModuleDict(
                {
                    action_type: nn.ModuleDict(
                        {
                            parameter_name: nn.Linear(feature_dim, parameter_size)
                            for parameter_name, parameter_size in head_config.items()
                        }
                    )
                    for action_type, head_config in param_heads.items()
                }
            )

        def forward(self, inputs):
            features = self.trunk(inputs)
            return {
                "action_type": self.action_head(features),
                "parameters": {
                    action_type: {
                        parameter_name: layer(features)
                        for parameter_name, layer in action_heads.items()
                    }
                    for action_type, action_heads in self.param_heads.items()
                },
            }

    return PolicyNetwork()


def _masked_argmax(torch_module: Any, logits: Any, mask: Any) -> int:
    if not bool(mask.any().item()):
        raise RuntimeError("PPO runtime could not find any valid choices for the current head")
    masked_logits = logits.clone()
    masked_logits[~mask] = torch_module.finfo(masked_logits.dtype).min
    return int(torch_module.argmax(masked_logits).item())


def _mask_for_named_choices(torch_module: Any, names: tuple[str, ...], allowed_names: list[str], logits: Any):
    mask = torch_module.zeros(logits.shape[-1], dtype=torch_module.bool, device=logits.device)
    for index, name in enumerate(names):
        if index >= mask.shape[0]:
            break
        if name in allowed_names:
            mask[index] = True
    return mask


def _mask_for_count(torch_module: Any, count: int, logits: Any):
    mask = torch_module.zeros(logits.shape[-1], dtype=torch_module.bool, device=logits.device)
    for index in range(min(count, mask.shape[0])):
        mask[index] = True
    return mask


@dataclass
class PolicyRuntime:
    checkpoint_path: str | Path
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.torch = _require_torch()
        checkpoint_path = Path(self.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        payload = self.torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(payload, dict):
            raise CheckpointIncompatibleError("checkpoint payload must be a dict")
        validate_checkpoint_payload(payload)

        self.action_types = tuple(payload["action_types"])
        self.param_heads = payload["param_heads"]
        self.architecture = payload["architecture"]
        self.model = build_policy_network(
            self.torch,
            obs_dim=payload["obs_dim"],
            action_types=self.action_types,
            architecture=self.architecture,
            param_heads=self.param_heads,
        )
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _select_action_type(self, logits: Any, env: ReceiptExtractionEnv) -> str:
        mask_map = action_type_mask(env)
        mask = self.torch.tensor([mask_map[action_type] for action_type in self.action_types], dtype=self.torch.bool, device=logits.device)
        action_index = _masked_argmax(self.torch, logits, mask)
        return self.action_types[action_index]

    def _select_field(self, logits: Any, allowed_fields: list[str]) -> str:
        mask = _mask_for_named_choices(self.torch, FIELD_ORDER, allowed_fields, logits)
        field_index = _masked_argmax(self.torch, logits, mask)
        return FIELD_ORDER[field_index]

    def _select_window(self, logits: Any, env: ReceiptExtractionEnv) -> str:
        allowed_windows = [window for window in WINDOW_ORDER if window in env.task.visible_windows]
        mask = _mask_for_named_choices(self.torch, WINDOW_ORDER, allowed_windows, logits)
        window_index = _masked_argmax(self.torch, logits, mask)
        return WINDOW_ORDER[window_index]

    def _select_bbox_id(self, logits: Any, env: ReceiptExtractionEnv) -> str:
        visible_ids = visible_region_ids(env)
        mask = _mask_for_count(self.torch, len(visible_ids), logits)
        return visible_ids[_masked_argmax(self.torch, logits, mask)]

    def _select_radius_bucket(self, logits: Any) -> int:
        mask = _mask_for_count(self.torch, len(RADIUS_BUCKETS), logits)
        radius_index = _masked_argmax(self.torch, logits, mask)
        return RADIUS_BUCKETS[radius_index]

    def _select_candidate_id(self, logits: Any, env: ReceiptExtractionEnv, field: str) -> str:
        candidates = list(env.last_observation.candidate_lists.get(field, []))
        mask = _mask_for_count(self.torch, len(candidates), logits)
        candidate_index = _masked_argmax(self.torch, logits, mask)
        return candidates[candidate_index].candidate_id

    def _select_line_item_candidate_id(self, logits: Any, env: ReceiptExtractionEnv) -> str:
        candidates = list(env.last_observation.line_item_candidates)
        mask = _mask_for_count(self.torch, len(candidates), logits)
        candidate_index = _masked_argmax(self.torch, logits, mask)
        return candidates[candidate_index].candidate_id

    def _select_line_item_index(self, logits: Any, env: ReceiptExtractionEnv) -> int:
        line_items = list(env.last_observation.current_draft.line_items)
        mask = _mask_for_count(self.torch, len(line_items), logits)
        return _masked_argmax(self.torch, logits, mask)

    def decode_action(self, action_type: str, parameter_outputs: dict[str, Any], env: ReceiptExtractionEnv) -> ReceiptAction:
        if action_type == "view_receipt":
            return ReceiptAction(action_type=action_type)
        if action_type == "list_text_regions":
            return ReceiptAction(action_type=action_type, window=self._select_window(parameter_outputs["window"][0], env))
        if action_type == "inspect_bbox":
            return ReceiptAction(action_type=action_type, bbox_id=self._select_bbox_id(parameter_outputs["bbox_index"][0], env))
        if action_type == "inspect_neighbors":
            return ReceiptAction(
                action_type=action_type,
                bbox_id=self._select_bbox_id(parameter_outputs["bbox_index"][0], env),
                radius_bucket=self._select_radius_bucket(parameter_outputs["radius_bucket"][0]),
            )
        if action_type == "query_candidates":
            field = self._select_field(parameter_outputs["field"][0], queryable_fields(env))
            return ReceiptAction(action_type=action_type, field=field)
        if action_type == "set_field_from_candidate":
            field = self._select_field(parameter_outputs["field"][0], candidate_fields(env))
            candidate_id = self._select_candidate_id(parameter_outputs["candidate_index"][0], env, field)
            return ReceiptAction(action_type=action_type, field=field, candidate_id=candidate_id)
        if action_type == "query_line_item_candidates":
            return ReceiptAction(action_type=action_type)
        if action_type == "add_line_item_from_candidate":
            return ReceiptAction(action_type=action_type, candidate_id=self._select_line_item_candidate_id(parameter_outputs["candidate_index"][0], env))
        if action_type == "remove_line_item":
            return ReceiptAction(action_type=action_type, line_item_index=self._select_line_item_index(parameter_outputs["line_item_index"][0], env))
        if action_type == "normalize_field":
            return ReceiptAction(action_type=action_type, field=self._select_field(parameter_outputs["field"][0], filled_fields(env)))
        if action_type == "check_total_consistency":
            return ReceiptAction(action_type=action_type)
        if action_type == "check_date_format":
            return ReceiptAction(action_type=action_type)
        if action_type == "check_receipt_consistency":
            return ReceiptAction(action_type=action_type)
        if action_type == "clear_field":
            return ReceiptAction(action_type=action_type, field=self._select_field(parameter_outputs["field"][0], filled_fields(env)))
        if action_type == "submit":
            return ReceiptAction(action_type=action_type)
        raise RuntimeError(f"unsupported PPO action type {action_type}")

    def select_action(self, env: ReceiptExtractionEnv) -> ReceiptAction:
        encoded = encode_observation(env.last_observation, env.task).to(self.device).unsqueeze(0)
        with self.torch.no_grad():
            outputs = self.model(encoded)
        action_type = self._select_action_type(outputs["action_type"][0], env)
        parameter_outputs = outputs["parameters"].get(action_type, {})
        return self.decode_action(action_type, parameter_outputs, env)


@dataclass
class PPOAgent:
    checkpoint_path: str | Path
    device: str = "cpu"
    name: str = "ppo"

    def __post_init__(self) -> None:
        self.runtime = PolicyRuntime(checkpoint_path=self.checkpoint_path, device=self.device)

    def select_action(self, env: ReceiptExtractionEnv) -> ReceiptAction:
        return self.runtime.select_action(env)
