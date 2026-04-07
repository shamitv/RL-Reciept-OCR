from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from env.image_store import (
    ImageStoreError,
    decode_image_json_bytes,
    image_id_from_annotation_path,
    image_json_path_for_id,
    image_json_to_data_url,
    load_image_json_asset,
    write_image_json_asset,
)
from scripts.convert_images_to_json import convert_images


def test_write_and_decode_image_json_asset(tmp_path: Path) -> None:
    image_path = tmp_path / "receipt.jpg"
    image_path.write_bytes(b"image-bytes")

    output_path = write_image_json_asset(image_path, tmp_path / "img_json")
    asset = load_image_json_asset(output_path)
    decoded, media_type = decode_image_json_bytes(output_path)

    assert asset.image_id == "receipt.jpg"
    assert asset.mime_type == "image/jpeg"
    assert decoded == b"image-bytes"
    assert media_type == "image/jpeg"
    assert image_json_to_data_url(output_path) == f"data:image/jpeg;base64,{asset.image_data}"


def test_image_json_validation_rejects_invalid_base64(tmp_path: Path) -> None:
    path = tmp_path / "bad.jpg.json"
    path.write_text(
        json.dumps({"image_id": "bad.jpg", "mime_type": "image/jpeg", "image_data": "not base64"}),
        encoding="utf-8",
    )

    with pytest.raises(ImageStoreError, match="Invalid base64"):
        load_image_json_asset(path)


def test_image_json_paths_derive_from_annotation_name(tmp_path: Path) -> None:
    annotation_path = tmp_path / "ann" / "odd-receipt.jpg.png.json"

    image_id = image_id_from_annotation_path(annotation_path)
    image_json_path = image_json_path_for_id(tmp_path, image_id)

    assert image_id == "odd-receipt.jpg.png"
    assert image_json_path == tmp_path / "img_json" / "odd-receipt.jpg.png.json"


def test_convert_images_to_json_script_function(tmp_path: Path) -> None:
    image_dir = tmp_path / "img"
    image_dir.mkdir()
    (image_dir / "a.jpg").write_bytes(b"a")
    (image_dir / "b.png").write_bytes(b"b")

    summary = convert_images(tmp_path)

    assert summary["source_images"] == 2
    assert summary["written"] == 2
    assert summary["invalid"] == 0
    assert decode_image_json_bytes(tmp_path / "img_json" / "a.jpg.json")[0] == b"a"
    assert json.loads((tmp_path / "img_json" / "b.png.json").read_text(encoding="utf-8"))["image_data"] == base64.b64encode(b"b").decode("ascii")


def test_convert_images_validates_existing_json_when_source_images_removed(tmp_path: Path) -> None:
    image_dir = tmp_path / "img"
    image_dir.mkdir()
    (image_dir / "a.jpg").write_bytes(b"a")
    convert_images(tmp_path)
    (image_dir / "a.jpg").unlink()
    image_dir.rmdir()

    summary = convert_images(tmp_path)

    assert summary["source_images"] == 0
    assert summary["written"] == 0
    assert summary["existing"] == 1
    assert summary["invalid"] == 0
