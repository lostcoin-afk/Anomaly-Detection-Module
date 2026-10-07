#!/usr/bin/env python3
"""Versioned, auditable threshold calibration for one PatchCore model.

This script deliberately uses the SAME PatchCoreEngine path that production
inference uses. It does not use Anomalib Engine.predict, because calibration
must be performed in the score space consumed by the application.

Default behavior:
  * scores validation/good images
  * computes the requested quantile threshold
  * writes an immutable calibration record
  * DOES NOT change model.json

Use --apply only after reviewing the calibration record.

Example:
  python scripts/calibrate_patchcore.py \
      --product harting_baseline_10e_m_s \
      --view top \
      --quantile 0.995

Then, after reviewing the saved JSON:
  python scripts/calibrate_patchcore.py \
      --product harting_baseline_10e_m_s \
      --view top \
      --quantile 0.995 \
      --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from harting_anomaly.core.config import CONFIG
from harting_anomaly.data.catalog import normalize_identifier
from harting_anomaly.models.patchcore_engine import PatchCoreEngine

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SCHEMA_VERSION = 1


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def numeric_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "p95": None,
            "p99": None,
            "p99_5": None,
            "p99_9": None,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "p99_5": float(np.quantile(arr, 0.995)),
        "p99_9": float(np.quantile(arr, 0.999)),
    }


def manifest_hash(files: list[Path]) -> str:
    h = hashlib.sha256()
    for path in files:
        h.update(str(path.resolve()).encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(path).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate a PatchCore threshold from known-good validation images.")
    parser.add_argument("--product", required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--quantile", type=float, default=None, help="Threshold quantile, e.g. 0.995")
    parser.add_argument("--validation-max", type=int, default=0, help="0 = all validation images")
    parser.add_argument("--apply", action="store_true", help="Also update model.json with the new threshold")
    parser.add_argument("--output-dir", default=None, help="Optional calibration directory override")
    args = parser.parse_args()

    product = normalize_identifier(args.product)
    view = normalize_identifier(args.view)

    model_dir = CONFIG.paths.patchcore_models_root / product / view
    dataset_dir = CONFIG.paths.prepared_data_root / product / view
    model_json_path = model_dir / "model.json"
    checkpoint_path = model_dir / "model.ckpt"
    val_dir = dataset_dir / "val" / "good"

    for required in (model_json_path, checkpoint_path, val_dir):
        if not required.exists():
            raise FileNotFoundError(f"Missing required artifact: {required}")

    metadata_before = load_json(model_json_path)
    configured_quantile = float(metadata_before.get("threshold_quantile", 0.995))
    quantile = configured_quantile if args.quantile is None else float(args.quantile)

    if not 0.5 < quantile < 1.0:
        raise ValueError("--quantile must be between 0.5 and 1.0")

    files = image_files(val_dir)
    if args.validation_max > 0:
        files = files[: args.validation_max]

    if len(files) < 5:
        raise RuntimeError(
            f"Only {len(files)} validation images found. Refusing to calibrate with fewer than 5 images."
        )

    # Production score path.
    engine = PatchCoreEngine.load(model_dir, device=args.device)

    rows: list[dict[str, Any]] = []
    scores: list[float] = []

    print("PATCHCORE THRESHOLD CALIBRATION")
    print("=" * 88)
    print(f"Product        : {product}")
    print(f"View           : {view}")
    print(f"Device         : {engine.device}")
    print(f"Validation dir : {val_dir}")
    print(f"Images         : {len(files)}")
    print(f"Quantile       : {quantile}")
    print(f"Apply          : {args.apply}")
    print()

    for index, path in enumerate(files, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode validation image: {path}")

        result = engine.predict(image)
        score = float(result.score)
        if not math.isfinite(score):
            raise RuntimeError(f"Non-finite PatchCore score for {path}: {score}")

        scores.append(score)
        rows.append(
            {
                "index": index - 1,
                "path": str(path.resolve()),
                "score": score,
                "image_shape": list(image.shape),
            }
        )

        if index == 1 or index == len(files) or index % 25 == 0:
            print(f"  scored {index:>4}/{len(files)}")

    threshold = float(np.quantile(np.asarray(scores, dtype=np.float64), quantile))
    above = int(np.sum(np.asarray(scores) > threshold))
    equal = int(np.sum(np.isclose(np.asarray(scores), threshold, rtol=0.0, atol=1e-12)))
    fpr = float(above / len(scores))

    calibration_id = f"calibration_{timestamp_slug()}"
    calibration_dir = (
        Path(args.output_dir)
        if args.output_dir
        else model_dir / "calibration"
    )
    calibration_path = calibration_dir / f"{calibration_id}.json"
    latest_path = calibration_dir / "latest.json"

    checkpoint_sha256 = sha256_file(checkpoint_path)
    validation_manifest_sha256 = manifest_hash(files)

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "calibration_id": calibration_id,
        "created_at": now_iso(),
        "product": product,
        "view": view,
        "model_dir": str(model_dir.resolve()),
        "model_checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "score_space": {
            "source": "harting_anomaly.models.patchcore_engine.PatchCoreEngine",
            "raw_pred_score": True,
            "classification_rule": "score > threshold",
        },
        "preprocessing": {
            "rgb": True,
            "scale": "uint8_to_float32_[0,1]",
            "resize": list(engine.image_size),
            "imagenet_normalize": True,
            "preprocessing_applied_once": True,
        },
        "calibration_dataset": {
            "split": "val/good",
            "directory": str(val_dir.resolve()),
            "count": len(files),
            "validation_manifest_sha256": validation_manifest_sha256,
            "all_samples_expected_good": True,
        },
        "method": {
            "name": "validation_quantile",
            "quantile": quantile,
            "threshold": threshold,
            "rule": f"threshold = quantile(scores, {quantile})",
        },
        "score_statistics": numeric_stats(scores),
        "false_positive_diagnostic": {
            "count_above_threshold": above,
            "count_equal_threshold": equal,
            "observed_fraction_above_threshold": fpr,
        },
        "samples": rows,
        "model_json_before": {
            "threshold": metadata_before.get("threshold"),
            "threshold_method": metadata_before.get("threshold_method"),
            "threshold_quantile": metadata_before.get("threshold_quantile"),
            "validation_count": metadata_before.get("validation_count"),
            "calibration_id": metadata_before.get("calibration_id"),
        },
        "applied": False,
    }

    write_json(calibration_path, record)
    write_json(latest_path, {**record, "latest_pointer": calibration_path.name})

    if args.apply:
        # Refuse to apply if the checkpoint changed during calibration.
        current_sha256 = sha256_file(checkpoint_path)
        if current_sha256 != checkpoint_sha256:
            raise RuntimeError("Checkpoint changed during calibration; refusing to update model.json")

        metadata_after = dict(metadata_before)
        metadata_after["threshold"] = threshold
        metadata_after["threshold_method"] = "validation_quantile"
        metadata_after["threshold_quantile"] = quantile
        metadata_after["validation_count"] = len(files)
        metadata_after["calibration_id"] = calibration_id
        metadata_after["calibration_created_at"] = record["created_at"]
        metadata_after["calibration_file"] = str(calibration_path.resolve())
        metadata_after["calibration_checkpoint_sha256"] = checkpoint_sha256
        metadata_after["calibration_score_space"] = "PatchCoreEngine.raw_pred_score"

        # Atomic replacement of model.json.
        tmp_path = model_json_path.with_suffix(".json.tmp")
        write_json(tmp_path, metadata_after)
        os.replace(tmp_path, model_json_path)

        record["applied"] = True
        record["applied_at"] = now_iso()
        record["model_json_after"] = {
            "threshold": threshold,
            "threshold_method": "validation_quantile",
            "threshold_quantile": quantile,
            "validation_count": len(files),
            "calibration_id": calibration_id,
        }
        write_json(calibration_path, record)
        write_json(latest_path, {**record, "latest_pointer": calibration_path.name})

    print()
    print("RESULT")
    print("-" * 88)
    print(f"Old threshold : {metadata_before.get('threshold')}")
    print(f"New threshold : {threshold}")
    print(f"Score min/max : {min(scores)} / {max(scores)}")
    print(f"Observed > threshold: {above}/{len(scores)} = {fpr:.2%}")
    print(f"Calibration file: {calibration_path}")
    print(f"Latest file    : {latest_path}")
    print(f"Applied        : {record['applied']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())