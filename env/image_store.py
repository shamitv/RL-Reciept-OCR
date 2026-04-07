from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IMAGE_JSON_DIR_NAME = "img_json"


class ImageStoreError(ValueError):
    """Raised when an image JSON asset is missing or malformed."""


@dataclass(frozen=True)
class ImageJsonAsset:
    image_id: str
    mime_type: str
    image_data: str
    path: Path | None = None

    def decode_bytes(self) -> bytes:
        try:
            return base64.b64decode(self.image_data, validate=True)
        except Exception as exc:  # pragma: no cover - exact binascii wording varies
            location = f" at {self.path}" if self.path else ""
            raise ImageStoreError(f"Invalid base64 image_data{location}") from exc

    def data_url(self) -> str:
        self.decode_bytes()
        return f"data:{self.mime_type};base64,{self.image_data}"


def image_id_from_annotation_path(annotation_path: str | Path) -> str:
    path = Path(annotation_path)
    if not path.name.endswith(".json"):
        raise ImageStoreError(f"Annotation path must end with .json: {path}")
    return path.name[:-5]


def image_json_path_for_id(dataset_root: str | Path, image_id: str) -> Path:
    return Path(dataset_root) / IMAGE_JSON_DIR_NAME / f"{image_id}.json"


def guess_image_mime_type(image_id: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_id)
    if mime_type and mime_type.startswith("image/"):
        return mime_type
    return "application/octet-stream"


def build_image_json_payload(image_path: str | Path) -> dict[str, str]:
    path = Path(image_path)
    return {
        "image_id": path.name,
        "mime_type": guess_image_mime_type(path.name),
        "image_data": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def write_image_json_asset(image_path: str | Path, output_dir: str | Path, *, overwrite: bool = False) -> Path:
    source_path = Path(image_path)
    output_path = Path(output_dir) / f"{source_path.name}.json"
    if output_path.exists() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_image_json_payload(source_path)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_image_json_asset(image_json_path: str | Path) -> ImageJsonAsset:
    path = Path(image_json_path)
    if not path.exists():
        raise ImageStoreError(f"Image JSON asset not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ImageStoreError(f"Image JSON asset is not valid JSON: {path}") from exc
    return image_json_asset_from_payload(payload, path=path)


def image_json_asset_from_payload(payload: Any, *, path: Path | None = None) -> ImageJsonAsset:
    if not isinstance(payload, dict):
        location = f": {path}" if path else ""
        raise ImageStoreError(f"Image JSON asset must be an object{location}")

    image_id = payload.get("image_id")
    mime_type = payload.get("mime_type")
    image_data = payload.get("image_data")
    if not isinstance(image_id, str) or not image_id.strip():
        raise ImageStoreError(f"Image JSON asset has invalid image_id: {path}")
    if not isinstance(mime_type, str) or not mime_type.strip():
        raise ImageStoreError(f"Image JSON asset has invalid mime_type: {path}")
    if not isinstance(image_data, str) or not image_data.strip():
        raise ImageStoreError(f"Image JSON asset has invalid image_data: {path}")

    asset = ImageJsonAsset(
        image_id=image_id.strip(),
        mime_type=mime_type.strip(),
        image_data="".join(image_data.split()),
        path=path,
    )
    asset.decode_bytes()
    return asset


def image_json_to_data_url(image_json_path: str | Path) -> str:
    return load_image_json_asset(image_json_path).data_url()


def decode_image_json_bytes(image_json_path: str | Path) -> tuple[bytes, str]:
    asset = load_image_json_asset(image_json_path)
    return asset.decode_bytes(), asset.mime_type
