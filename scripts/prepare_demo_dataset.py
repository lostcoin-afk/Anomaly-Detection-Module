from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from harting_anomaly.core.config import CONFIG
from harting_anomaly.data.catalog import normalize_identifier, normalize_view


SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_product_dir(product: str) -> Path:
    target = normalize_identifier(product)
    for path in CONFIG.paths.raw_data_root.iterdir():
        if path.is_dir() and normalize_identifier(path.name) == target:
            return path
    raise FileNotFoundError(f"Product not found under {CONFIG.paths.raw_data_root}: {product}")


def find_view_dir(product_dir: Path, view: str) -> Path:
    requested = normalize_identifier(view)
    for path in product_dir.iterdir():
        if path.is_dir() and normalize_identifier(normalize_view(path.name)) == requested:
            return path
    raise FileNotFoundError(f"View '{view}' not found in {product_dir}")


def split_for(path: Path, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.sha1(str(path).encode(), usedforsecurity=False).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    product_dir = find_product_dir(args.product)
    view_dir = find_view_dir(product_dir, args.view)
    product_id = normalize_identifier(args.product)
    view_id = normalize_identifier(args.view)

    output_root = CONFIG.paths.prepared_data_root / product_id / view_id

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)

    for split in ("train", "val", "test"):
        (output_root / split / "good").mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in view_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    if len(images) < 10:
        raise RuntimeError(f"Only {len(images)} images found; need more for a useful prototype.")

    train_ratio = CONFIG.dataset.train_ratio
    val_ratio = CONFIG.dataset.validation_ratio

    counts = {"train": 0, "val": 0, "test": 0}
    for source in images:
        split = split_for(source, train_ratio, val_ratio)
        destination = output_root / split / "good" / source.name
        if destination.exists():
            destination = output_root / split / "good" / f"{source.stem}_{abs(hash(source))}{source.suffix}"
        shutil.copy2(source, destination)
        counts[split] += 1

    print(f"Prepared: {product_id} / {view_id}")
    print(f"Source : {view_dir}")
    print(f"Output : {output_root}")
    print(f"Counts : {counts}")
    print("\nPrototype note: this split is image-level. Replace it with piece/session grouping once the filename/acquisition scheme is known.")


if __name__ == "__main__":
    main()