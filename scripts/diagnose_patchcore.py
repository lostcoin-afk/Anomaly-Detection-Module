#!/usr/bin/env python3
"""Read-only PatchCore diagnostic for one (product, view) model.

The important change from the previous diagnostic is the inference-path test.
Anomalib v2.6.2 applies its PreProcessor during the Lightning prediction
callback. Therefore, passing ``model.pre_processor`` as PredictDataset's
transform causes DOUBLE preprocessing and can produce a huge score spike.

For a fair Engine comparison this script instead:
  1. constructs Patchcore(pre_processor=False)
  2. applies the configured preprocessor exactly once as the dataset transform
  3. runs Engine.predict()

It also compares a second independent reference path using PredictDataset's
RGB loader + the same preprocessor + model.model, and it fixes image_path
handling for batched Anomalib predictions.

Usage:
  python scripts/diagnose_patchcore.py \
      --product harting_baseline_10e_m_s \
      --view top \
      --device cuda

Optional runtime probe:
  python scripts/diagnose_patchcore.py \
      --product harting_baseline_10e_m_s \
      --view top \
      --device cuda \
      --source 0 \
      --runtime-frames 10

This script never modifies model.ckpt, model.json, or the dataset.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import torch

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

from harting_anomaly.core.config import CONFIG
from harting_anomaly.data.catalog import normalize_identifier
from harting_anomaly.models.patchcore_engine import PatchCoreEngine

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DiagnosisContext:
    product: str
    view: str
    model_dir: Path
    dataset_dir: Path
    checkpoint_path: Path
    metadata_path: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return super().default(obj)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    t = tensor.detach().float().cpu()
    finite = torch.isfinite(t)
    result: dict[str, Any] = {
        "shape": list(t.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(t.numel()),
        "finite_fraction": float(finite.float().mean().item()) if t.numel() else 1.0,
    }
    if t.numel():
        result.update(
            {
                "min": float(t.min().item()),
                "max": float(t.max().item()),
                "mean": float(t.mean().item()),
                "std": float(t.std(unbiased=False).item()),
                "abs_max": float(t.abs().max().item()),
            }
        )
    return result


def numeric_stats(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "p99_5": None,
            "p99_9": None,
        }

    arr = np.asarray(clean, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "p99_5": float(np.quantile(arr, 0.995)),
        "p99_9": float(np.quantile(arr, 0.999)),
    }


def correlation_stats(a: list[float], b: list[float]) -> dict[str, Any]:
    n = min(len(a), len(b))
    if n == 0:
        return {"count": 0}

    aa = np.asarray(a[:n], dtype=np.float64)
    bb = np.asarray(b[:n], dtype=np.float64)
    diff = aa - bb
    denom = max(float(np.max(np.abs(aa))), float(np.max(np.abs(bb))), 1e-12)
    pearson = None
    if n > 1 and np.std(aa) > 0 and np.std(bb) > 0:
        pearson = float(np.corrcoef(aa, bb)[0, 1])

    return {
        "count": int(n),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "max_relative_to_score_scale": float(np.max(np.abs(diff)) / denom),
        "pearson": pearson,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_context(product: str, view: str) -> DiagnosisContext:
    product = normalize_identifier(product)
    view = normalize_identifier(view)
    model_dir = CONFIG.paths.patchcore_models_root / product / view
    dataset_dir = CONFIG.paths.prepared_data_root / product / view
    return DiagnosisContext(
        product=product,
        view=view,
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        checkpoint_path=model_dir / "model.ckpt",
        metadata_path=model_dir / "model.json",
        train_dir=dataset_dir / "train" / "good",
        val_dir=dataset_dir / "val" / "good",
        test_dir=dataset_dir / "test" / "good",
    )


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_metadata(context: DiagnosisContext) -> dict[str, Any]:
    return json.loads(context.metadata_path.read_text(encoding="utf-8"))


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return requested


def create_patchcore_from_metadata(
    metadata: dict[str, Any],
    *,
    pre_processor: bool,
    post_processor: bool = False,
) -> Patchcore:
    image_size = tuple(metadata.get("image_size", [256, 256]))
    configured_preprocessor = (
        Patchcore.configure_pre_processor(image_size=image_size)
        if pre_processor
        else False
    )
    return Patchcore(
        backbone=metadata["backbone"],
        layers=tuple(metadata["layers"]),
        pre_trained=False,
        coreset_sampling_ratio=float(metadata["coreset_sampling_ratio"]),
        num_neighbors=int(metadata["num_neighbors"]),
        pre_processor=configured_preprocessor,
        post_processor=post_processor,
        evaluator=False,
        visualizer=False,
    )


def load_checkpoint_state(context: DiagnosisContext) -> dict[str, Any]:
    try:
        checkpoint = torch.load(
            context.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(context.checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(checkpoint)!r}")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint state_dict is not a dictionary")
    return {str(k): v for k, v in state_dict.items()}


def load_model_for_diagnosis(
    context: DiagnosisContext,
    metadata: dict[str, Any],
    device: str,
) -> tuple[Patchcore, dict[str, Any]]:
    model = create_patchcore_from_metadata(metadata, pre_processor=True, post_processor=False)
    state_dict = load_checkpoint_state(context)
    incompat = model.load_state_dict(state_dict, strict=False)
    missing = [str(x) for x in incompat.missing_keys]
    unexpected = [str(x) for x in incompat.unexpected_keys]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint architecture/state mismatch. "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    return model, {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "strict_load_clean": True,
        "device": str(torch_device),
    }


def get_memory_bank_report(model: Patchcore) -> dict[str, Any]:
    memory_bank = model.model.memory_bank
    result = tensor_stats(memory_bank)
    result["is_empty"] = bool(memory_bank.numel() == 0 or memory_bank.shape[0] == 0)
    result["rows"] = int(memory_bank.shape[0]) if memory_bank.ndim >= 1 else 0
    result["embedding_dimension"] = int(memory_bank.shape[1]) if memory_bank.ndim >= 2 else None
    return result


def preprocess_cv2_with_model(
    model: Patchcore,
    image_bgr: np.ndarray,
    device: str,
) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().div(255.0)
    tensor = tensor.unsqueeze(0).to(torch.device(device))
    return model.pre_processor(tensor)


@torch.inference_mode()
def direct_model_score(model: Patchcore, tensor: torch.Tensor) -> dict[str, Any]:
    prediction = model.model(tensor)
    return {
        "score": float(prediction.pred_score.detach().float().cpu().reshape(-1)[0].item()),
        "anomaly_map_stats": numeric_stats(
            prediction.anomaly_map.detach().float().cpu().numpy().reshape(-1).tolist()
        ),
    }


def score_manual_predictdataset_reference(
    context: DiagnosisContext,
    image_path: Path,
    metadata: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Reference path: Anomalib RGB loader -> same preprocessor -> model.model.

    This bypasses Lightning/Engine entirely and is useful for isolating input
    decoding/preprocessing/model issues.
    """
    model = create_patchcore_from_metadata(metadata, pre_processor=True, post_processor=False)
    state_dict = load_checkpoint_state(context)
    model.load_state_dict(state_dict, strict=True)
    model.to(torch.device(device))
    model.eval()

    dataset = PredictDataset(path=image_path)
    item = dataset[0]
    raw = item.image.unsqueeze(0).to(torch.device(device))
    preprocessed = model.pre_processor(raw)
    result = direct_model_score(model, preprocessed)

    return {
        "path": str(image_path),
        "raw_rgb_tensor": tensor_stats(raw),
        "preprocessed_tensor": tensor_stats(preprocessed),
        "score": result["score"],
        "anomaly_map_stats": result["anomaly_map_stats"],
        "image_path_from_dataset": str(item.image_path),
    }


def score_engine_equivalent(
    context: DiagnosisContext,
    image_path: Path,
    metadata: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Run Engine.predict with exactly ONE preprocessing stage.

    PredictDataset applies its ``transform`` and the model is constructed with
    ``pre_processor=False``. This prevents the double-preprocessing bug caused
    by using model.pre_processor both as a dataset transform and as a Lightning
    callback.
    """
    preprocessor = Patchcore.configure_pre_processor(
        image_size=tuple(metadata.get("image_size", [256, 256]))
    )
    model = create_patchcore_from_metadata(metadata, pre_processor=False, post_processor=False)
    state_dict = load_checkpoint_state(context)
    model.load_state_dict(state_dict, strict=True)

    accelerator = "gpu" if device.startswith("cuda") else "cpu"
    engine = Engine(accelerator=accelerator, devices=1, logger=False)
    dataset = PredictDataset(path=image_path, transform=preprocessor)

    started = time.perf_counter()
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(context.checkpoint_path),
        return_predictions=True,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if not predictions:
        raise RuntimeError("Engine.predict returned no predictions")

    pred = predictions[0]
    score = float(pred.pred_score.detach().float().cpu().reshape(-1)[0].item())
    image_paths = extract_prediction_paths(pred)

    return {
        "path": str(image_path),
        "score": score,
        "elapsed_api_ms": float(elapsed_ms),
        "image_paths": image_paths,
        "pred_label": (
            int(pred.pred_label.detach().cpu().reshape(-1)[0].item())
            if pred.pred_label is not None
            else None
        ),
        "anomaly_map_shape": list(pred.anomaly_map.shape) if pred.anomaly_map is not None else None,
    }


def extract_prediction_paths(prediction: Any) -> list[str]:
    value = getattr(prediction, "image_path", None)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    if isinstance(value, np.ndarray):
        return [str(x) for x in value.reshape(-1).tolist()]
    return [str(value)]


def score_validation_engine(
    context: DiagnosisContext,
    metadata: dict[str, Any],
    device: str,
    max_images: int,
) -> tuple[list[tuple[str, float]], dict[str, Any]]:
    files = image_files(context.val_dir)
    if max_images > 0:
        files = files[:max_images]
    if not files:
        return [], {"count": 0, "error": "No validation images found"}

    preprocessor = Patchcore.configure_pre_processor(
        image_size=tuple(metadata.get("image_size", [256, 256]))
    )
    model = create_patchcore_from_metadata(metadata, pre_processor=False, post_processor=False)
    state_dict = load_checkpoint_state(context)
    model.load_state_dict(state_dict, strict=True)

    accelerator = "gpu" if device.startswith("cuda") else "cpu"
    engine = Engine(accelerator=accelerator, devices=1, logger=False)
    dataset = PredictDataset(path=context.val_dir, transform=preprocessor)

    started = time.perf_counter()
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=str(context.checkpoint_path),
        return_predictions=True,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    pairs: list[tuple[str, float]] = []
    for prediction in predictions or []:
        scores = prediction.pred_score.detach().float().cpu().reshape(-1).tolist()
        paths = extract_prediction_paths(prediction)
        if len(paths) != len(scores):
            # Fallback only for an unusual batch object. Keep the path count
            # mismatch visible rather than silently fabricating paths.
            raise RuntimeError(
                f"Prediction batch path/score mismatch: {len(paths)} paths vs {len(scores)} scores"
            )
        pairs.extend((str(Path(p).resolve()), float(s)) for p, s in zip(paths, scores, strict=True))

    return pairs[: len(files)], {
        "count": len(pairs[: len(files)]),
        "engine_predict_elapsed_ms": float(elapsed_ms),
        "engine_predict_ms_per_image": float(elapsed_ms / len(pairs)) if pairs else None,
        "preprocessing_mode": "PredictDataset.transform_only",
    }


def score_single_wrapper(wrapper: PatchCoreEngine, image_path: Path) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    result = wrapper.predict(image)
    return {
        "path": str(image_path.resolve()),
        "shape": list(image.shape),
        "score": float(result.score),
        "threshold": float(result.threshold),
        "is_anomaly": bool(result.is_anomaly),
        "inference_ms": float(result.inference_ms),
    }


def score_validation_wrapper(
    wrapper: PatchCoreEngine,
    files: list[Path],
) -> tuple[list[tuple[str, float]], dict[str, Any]]:
    scores: list[tuple[str, float]] = []
    started = time.perf_counter()
    for path in files:
        result = score_single_wrapper(wrapper, path)
        scores.append((result["path"], result["score"]))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return scores, {
        "count": len(scores),
        "wrapper_elapsed_ms": float(elapsed_ms),
        "wrapper_ms_per_image": float(elapsed_ms / len(scores)) if scores else None,
    }


def jpeg_roundtrip(wrapper: PatchCoreEngine, image_path: Path) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    )
    if not ok:
        raise RuntimeError("JPEG encode failed")
    roundtrip = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if roundtrip is None:
        raise RuntimeError("JPEG decode failed")

    original = wrapper.predict(image)
    jpeg = wrapper.predict(roundtrip)
    return {
        "original_score": float(original.score),
        "jpeg_roundtrip_score": float(jpeg.score),
        "absolute_score_delta": float(abs(original.score - jpeg.score)),
        "jpeg_quality": 80,
        "encoded_bytes": int(len(encoded)),
    }


@torch.inference_mode()
def patch_distance_probe(model: Patchcore, preprocessed: torch.Tensor) -> dict[str, Any]:
    core = model.model
    x = preprocessed.to(dtype=core.memory_bank.dtype)
    features = core.feature_extractor(x)
    features = {layer: core.feature_pooler(feature) for layer, feature in features.items()}
    embedding_4d = core.generate_embedding(features)
    batch_size, _, width, height = embedding_4d.shape
    embedding = core.reshape_embedding(embedding_4d)

    patch_scores, locations = core.nearest_neighbors(embedding=embedding, n_neighbors=1)
    patch_scores_b = patch_scores.reshape(batch_size, -1)
    locations_b = locations.reshape(batch_size, -1)
    model_score = core.compute_anomaly_score(patch_scores_b, locations_b, embedding)

    patch_values = patch_scores.detach().float().cpu().reshape(-1).tolist()
    return {
        "embedding_shape_before_reshape": list(embedding_4d.shape),
        "embedding_shape_after_reshape": list(embedding.shape),
        "embedding_stats": tensor_stats(embedding),
        "patch_distance_stats": numeric_stats(patch_values),
        "model_compute_anomaly_score": float(model_score.reshape(-1)[0].detach().float().cpu().item()),
        "num_neighbors": int(core.num_neighbors),
    }


def environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "anomalib": package_version("anomalib"),
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
            "timm": package_version("timm"),
            "opencv-python": package_version("opencv-python"),
            "numpy": package_version("numpy"),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        },
    }
    if torch.cuda.is_available():
        report["torch"]["devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "capability": list(torch.cuda.get_device_capability(i)),
                "total_memory_mb": round(
                    torch.cuda.get_device_properties(i).total_memory / 1024**2, 2
                ),
            }
            for i in range(torch.cuda.device_count())
        ]
    return report


def runtime_probe(wrapper: PatchCoreEngine, source: Any, frames: int) -> dict[str, Any]:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        return {"opened": False, "source": str(source), "error": "Could not open runtime source"}

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for index in range(max(1, frames)):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            result = wrapper.predict(frame)
            rows.append(
                {
                    "index": index,
                    "shape": list(frame.shape),
                    "score": float(result.score),
                    "is_anomaly": bool(result.is_anomaly),
                    "inference_ms": float(result.inference_ms),
                }
            )
    finally:
        capture.release()

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    scores = [x["score"] for x in rows]
    return {
        "opened": True,
        "source": str(source),
        "requested_frames": int(frames),
        "captured_frames": len(rows),
        "elapsed_ms": float(elapsed_ms),
        "frames": rows,
        "score_stats": numeric_stats(scores),
        "shape_set": sorted({tuple(x["shape"]) for x in rows}),
    }


def generate_conclusions(report: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    exact = report["exact_training_image"]
    wrapper_score = float(exact["wrapper"]["score"])
    reference_score = float(exact["reference"]["score"])
    engine_score = float(exact["anomalib_engine"]["score"])

    ref_diff = abs(wrapper_score - reference_score)
    engine_diff = abs(wrapper_score - engine_score)
    scale = max(abs(wrapper_score), abs(reference_score), abs(engine_score), 1e-9)

    if ref_diff > max(0.05, 0.01 * scale):
        out.append(
            {
                "severity": "HIGH",
                "message": f"Runtime wrapper differs from the RGB PredictDataset reference by {ref_diff:.6f}; inspect image decoding/channel order or preprocessing device placement.",
            }
        )
    else:
        out.append(
            {
                "severity": "OK",
                "message": f"Runtime wrapper agrees with the independent Anomalib RGB/preprocessor reference (delta={ref_diff:.6f}).",
            }
        )

    if engine_diff > max(0.05, 0.01 * scale):
        out.append(
            {
                "severity": "HIGH",
                "message": f"Engine-equivalent score still differs from the runtime wrapper by {engine_diff:.6f}; inspect Engine/dataset configuration before calibrating.",
            }
        )
    else:
        out.append(
            {
                "severity": "OK",
                "message": f"Engine-equivalent score agrees with runtime wrapper (delta={engine_diff:.6f}).",
            }
        )

    memory = report["memory_bank"]
    if memory["is_empty"]:
        out.append({"severity": "CRITICAL", "message": "PatchCore memory bank is empty."})
    else:
        out.append(
            {
                "severity": "OK",
                "message": f"Memory bank is populated ({memory['rows']} × {memory['embedding_dimension']}).",
            }
        )

    threshold_diag = report["threshold_diagnosis"]
    fpr = threshold_diag.get("wrapper_fraction_above_stored_threshold")
    if fpr is not None and fpr > 0.05:
        out.append(
            {
                "severity": "HIGH",
                "message": f"Stored threshold marks {fpr:.2%} of known-good validation images as anomalous in the runtime score space. Recalibration is required before trusting the classifier.",
            }
        )
    elif fpr is not None:
        out.append(
            {
                "severity": "OK",
                "message": f"Stored threshold marks {fpr:.2%} of known-good validation images as anomalous in the runtime score space.",
            }
        )

    stored = float(report["metadata"]["metadata"]["threshold"])
    p995 = report["validation"]["wrapper"]["stats"].get("p99_5")
    if p995 is not None:
        rel = abs(float(p995) - stored) / max(abs(stored), 1e-9)
        if rel > 0.02:
            out.append(
                {
                    "severity": "MEDIUM",
                    "message": f"Runtime P99.5 validation score ({p995:.6f}) differs from stored threshold ({stored:.6f}) by {rel:.2%}.",
                }
            )

    return out


def render_text(report: dict[str, Any]) -> str:
    lines = [
        "PATCHCORE V1 DIAGNOSTIC REPORT",
        "=" * 88,
        f"Generated: {report['generated_at']}",
        f"Product:   {report['context']['product']}",
        f"View:      {report['context']['view']}",
        "",
        "EXECUTIVE RESULT",
        "-" * 88,
    ]
    lines += [f"[{x['severity']}] {x['message']}" for x in report["conclusions"]]
    lines += ["", "EXACT TRAINING IMAGE", "-" * 88]
    exact = report["exact_training_image"]
    lines += [
        f"Path: {exact['path']}",
        f"Wrapper score: {exact['wrapper']['score']}",
        f"PredictDataset reference score: {exact['reference']['score']}",
        f"Engine-equivalent score: {exact['anomalib_engine']['score']}",
        f"Wrapper/reference delta: {exact['comparison']['wrapper_reference_abs_diff']}",
        f"Wrapper/Engine delta: {exact['comparison']['wrapper_engine_abs_diff']}",
        "",
        "MEMORY BANK",
        "-" * 88,
        str(report["memory_bank"]),
        "",
        "PREPROCESSING",
        "-" * 88,
        str(report["preprocessing"]),
        "",
        "PATCH DISTANCE PROBE",
        "-" * 88,
        str(report["patch_distance_probe"]),
        "",
        "VALIDATION",
        "-" * 88,
        f"Engine-equivalent: {report['validation']['anomalib']['stats']}",
        f"Wrapper:            {report['validation']['wrapper']['stats']}",
        "",
        "THRESHOLD DIAGNOSIS",
        "-" * 88,
        str(report["threshold_diagnosis"]),
        "",
        "WRAPPER VS ENGINE VALIDATION",
        "-" * 88,
        str(report["validation_comparison"]),
        "",
        "JPEG ROUNDTRIP",
        "-" * 88,
        str(report["jpeg_roundtrip"]),
        "",
        "ENVIRONMENT",
        "-" * 88,
        str(report["environment"]),
        "",
        "No training, checkpoint, dataset, or model metadata was modified.",
    ]
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only PatchCore diagnostic")
    parser.add_argument("--product", required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--validation-max", type=int, default=0, help="0 = all validation images")
    parser.add_argument("--source", default=None, help="Optional camera index, video, or RTSP path")
    parser.add_argument("--runtime-frames", type=int, default=10)
    parser.add_argument("--sample-train", type=int, default=3)
    parser.add_argument("--sample-test", type=int, default=3)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    started = time.perf_counter()
    context = discover_context(args.product, args.view)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir) if args.output_dir else CONFIG.paths.logs_root / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"patchcore_{context.product}__{context.view}_{stamp}"
    json_path = output_dir / f"{base}.json"
    txt_path = output_dir / f"{base}.txt"
    csv_path = output_dir / f"{base}_validation.csv"

    report: dict[str, Any] = {
        "generated_at": now_iso(),
        "script": str(Path(__file__).resolve()),
        "context": {
            "product": context.product,
            "view": context.view,
            "model_dir": str(context.model_dir),
            "dataset_dir": str(context.dataset_dir),
            "checkpoint_path": str(context.checkpoint_path),
            "metadata_path": str(context.metadata_path),
        },
        "requested_device": args.device,
        "resolved_device": device,
        "environment": environment_report(),
        "checkpoint": {
            "sha256": sha256_file(context.checkpoint_path),
            "size_mb": round(context.checkpoint_path.stat().st_size / 1024**2, 3),
        },
    }

    required = [
        context.model_dir,
        context.checkpoint_path,
        context.metadata_path,
        context.train_dir,
        context.val_dir,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))

    metadata = load_metadata(context)
    report["metadata"] = {
        "metadata": metadata,
        "required_fields_present": all(
            key in metadata
            for key in [
                "backbone",
                "layers",
                "coreset_sampling_ratio",
                "num_neighbors",
                "image_size",
                "threshold",
            ]
        ),
    }

    train_files = image_files(context.train_dir)
    val_files = image_files(context.val_dir)
    test_files = image_files(context.test_dir)
    report["dataset"] = {
        "train_files": len(train_files),
        "val_files": len(val_files),
        "test_files": len(test_files),
    }

    if not train_files or not val_files:
        raise RuntimeError("Training and validation images are required")

    if args.validation_max > 0:
        selected_val = val_files[: args.validation_max]
    else:
        selected_val = val_files

    exact_train = train_files[0]
    image = cv2.imread(str(exact_train), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode {exact_train}")

    print("PATCHCORE V1 DIAGNOSTICS")
    print("=" * 88)
    print(f"Product : {context.product}")
    print(f"View    : {context.view}")
    print(f"Device  : {device}")
    print()

    print("[1/7] Loading model/checkpoint...")
    model, load_info = load_model_for_diagnosis(context, metadata, device)
    report["checkpoint_load"] = load_info
    report["memory_bank"] = get_memory_bank_report(model)

    print("[2/7] Preprocessing and exact-image reference...")
    preprocessed = preprocess_cv2_with_model(model, image, device)
    report["preprocessing"] = {
        "source_image": str(exact_train.resolve()),
        "source_shape_bgr": list(image.shape),
        "preprocessed_cv2": tensor_stats(preprocessed),
    }

    wrapper = PatchCoreEngine.load(context.model_dir, device=device)
    wrapper_exact = score_single_wrapper(wrapper, exact_train)
    reference_exact = score_manual_predictdataset_reference(
        context, exact_train, metadata, device
    )
    engine_exact = score_engine_equivalent(
        context, exact_train, metadata, device
    )

    report["exact_training_image"] = {
        "path": str(exact_train.resolve()),
        "image_shape": list(image.shape),
        "wrapper": wrapper_exact,
        "reference": reference_exact,
        "anomalib_engine": engine_exact,
        "comparison": {
            "wrapper_reference_abs_diff": abs(
                wrapper_exact["score"] - reference_exact["score"]
            ),
            "wrapper_engine_abs_diff": abs(
                wrapper_exact["score"] - engine_exact["score"]
            ),
            "reference_engine_abs_diff": abs(
                reference_exact["score"] - engine_exact["score"]
            ),
        },
    }

    print("[3/7] Patch-level score probe...")
    report["patch_distance_probe"] = patch_distance_probe(model, preprocessed)
    report["patch_distance_probe"]["wrapper_exact_score"] = wrapper_exact["score"]

    print("[4/7] JPEG sensitivity...")
    report["jpeg_roundtrip"] = jpeg_roundtrip(wrapper, exact_train)

    print("[5/7] Validation scores...")
    engine_val, engine_meta = score_validation_engine(
        context,
        metadata,
        device,
        max_images=args.validation_max,
    )
    wrapper_val, wrapper_meta = score_validation_wrapper(wrapper, selected_val)

    report["validation"] = {
        "anomalib": {
            "stats": numeric_stats([score for _, score in engine_val]),
            "runtime": engine_meta,
        },
        "wrapper": {
            "stats": numeric_stats([score for _, score in wrapper_val]),
            "runtime": wrapper_meta,
        },
    }

    engine_map = {str(Path(path).resolve()): score for path, score in engine_val}
    wrapper_map = {str(Path(path).resolve()): score for path, score in wrapper_val}
    common = sorted(set(engine_map) & set(wrapper_map))
    engine_common = [engine_map[p] for p in common]
    wrapper_common = [wrapper_map[p] for p in common]
    report["validation_comparison"] = correlation_stats(wrapper_common, engine_common)
    report["validation_comparison"]["common_images"] = len(common)

    stored_threshold = float(metadata["threshold"])
    wrapper_scores = [score for _, score in wrapper_val]
    engine_scores = [score for _, score in engine_val]
    report["threshold_diagnosis"] = {
        "stored_threshold": stored_threshold,
        "wrapper_fraction_above_stored_threshold": (
            float(np.mean(np.asarray(wrapper_scores) > stored_threshold))
            if wrapper_scores
            else None
        ),
        "engine_fraction_above_stored_threshold": (
            float(np.mean(np.asarray(engine_scores) > stored_threshold))
            if engine_scores
            else None
        ),
        "wrapper_quantiles": {
            str(q): float(np.quantile(np.asarray(wrapper_scores), q))
            for q in [0.95, 0.99, 0.995, 0.999]
        } if wrapper_scores else {},
        "engine_quantiles": {
            str(q): float(np.quantile(np.asarray(engine_scores), q))
            for q in [0.95, 0.99, 0.995, 0.999]
        } if engine_scores else {},
        "metadata_threshold_method": metadata.get("threshold_method"),
        "metadata_threshold_quantile": metadata.get("threshold_quantile"),
        "metadata_validation_count": metadata.get("validation_count"),
    }

    train_samples = [score_single_wrapper(wrapper, p) for p in train_files[: max(0, args.sample_train)]]
    test_samples = [score_single_wrapper(wrapper, p) for p in test_files[: max(0, args.sample_test)]]
    report["extra_samples"] = {
        "training_images": train_samples,
        "test_images": test_samples,
    }

    if args.source is not None:
        print("[6/7] Runtime source probe...")
        source: Any = int(args.source) if args.source.isdigit() else args.source
        report["runtime_source"] = runtime_probe(
            wrapper, source, max(1, args.runtime_frames)
        )
    else:
        report["runtime_source"] = None

    print("[7/7] Writing report...")
    report["elapsed_seconds"] = time.perf_counter() - started
    report["conclusions"] = generate_conclusions(report)

    rows = []
    for path in common:
        rows.append(
            {
                "image_path": path,
                "engine_score": engine_map[path],
                "wrapper_score": wrapper_map[path],
                "absolute_diff": abs(engine_map[path] - wrapper_map[path]),
                "stored_threshold": stored_threshold,
                "wrapper_anomaly": wrapper_map[path] > stored_threshold,
                "engine_anomaly": engine_map[path] > stored_threshold,
            }
        )
    write_csv(csv_path, rows)

    json_path.write_text(
        json.dumps(report, indent=2, cls=NumpyJSONEncoder),
        encoding="utf-8",
    )
    txt_path.write_text(render_text(report), encoding="utf-8")

    print()
    print("=" * 88)
    print("DIAGNOSIS COMPLETE")
    print("=" * 88)
    for item in report["conclusions"]:
        print(f"[{item['severity']}] {item['message']}")
    print()
    print(f"JSON report : {json_path}")
    print(f"TXT report  : {txt_path}")
    print(f"CSV report  : {csv_path}")
    print(f"Elapsed     : {report['elapsed_seconds']:.2f} s")

    del model
    del wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())