from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.image_store import IMAGE_JSON_DIR_NAME, ImageStoreError, load_image_json_asset, write_image_json_asset


DEFAULT_DATASET_ROOT = ROOT / "dataset" / "Receipt dataset" / "ds0"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def image_files(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def validate_image_json_files(image_json_dir: Path) -> int:
    invalid = 0
    for image_json_path in sorted(image_json_dir.glob("*.json")):
        try:
            load_image_json_asset(image_json_path)
        except ImageStoreError:
            invalid += 1
    return invalid


def convert_images(dataset_root: str | Path, *, overwrite: bool = False) -> dict[str, int]:
    root = Path(dataset_root)
    image_dir = root / "img"
    image_json_dir = root / IMAGE_JSON_DIR_NAME
    if not image_dir.exists():
        existing_assets = sorted(image_json_dir.glob("*.json")) if image_json_dir.exists() else []
        if existing_assets:
            return {
                "source_images": 0,
                "written": 0,
                "existing": len(existing_assets),
                "invalid": validate_image_json_files(image_json_dir),
                "output_dir": str(image_json_dir),
            }
        raise FileNotFoundError(f"Image directory not found and no image JSON assets exist: {image_dir}")

    written = 0
    existing = 0
    invalid = 0
    for image_path in image_files(image_dir):
        output_path = image_json_dir / f"{image_path.name}.json"
        existed = output_path.exists()
        write_image_json_asset(image_path, image_json_dir, overwrite=overwrite)
        if existed and not overwrite:
            existing += 1
        else:
            written += 1
        try:
            load_image_json_asset(output_path)
        except ImageStoreError:
            invalid += 1

    return {
        "source_images": len(image_files(image_dir)),
        "written": written,
        "existing": existing,
        "invalid": invalid,
        "output_dir": str(image_json_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert receipt image files into per-image base64 JSON assets.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Dataset root containing img/.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate JSON assets even when they already exist.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = convert_images(args.dataset_root, overwrite=args.overwrite)
    print(
        "[CONVERT] "
        f"source_images={summary['source_images']} "
        f"written={summary['written']} "
        f"existing={summary['existing']} "
        f"invalid={summary['invalid']} "
        f"output_dir={summary['output_dir']}",
        flush=True,
    )
    return 1 if summary["invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
