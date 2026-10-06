from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from harting_anomaly.core.config import CONFIG
from harting_anomaly.models.patchcore_engine import PatchCoreEngine
from harting_anomaly.monitoring.system_monitor import SystemMonitor


@dataclass
class RuntimeSnapshot:
    frame_id: int
    product: str
    view: str
    score: float
    threshold: float
    status: str
    inference_ms: float
    fps: float
    system: dict
    raw_jpeg: bytes
    heatmap_jpeg: bytes
    overlay_jpeg: bytes
    composite_jpeg: bytes

    def public_dict(self) -> dict:
        data = asdict(self)
        data.pop("raw_jpeg", None)
        data.pop("heatmap_jpeg", None)
        data.pop("overlay_jpeg", None)
        data.pop("composite_jpeg", None)
        return data


class InferenceRuntime:
    def __init__(self) -> None:
        self.product = ""
        self.view = ""
        self._engine: PatchCoreEngine | None = None
        self._monitor = SystemMonitor(interval_seconds=0.5)
        self._capture = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: RuntimeSnapshot | None = None
        self._frame_id = 0
        self._last_frame_time = time.perf_counter()

    def start(self, product: str, view: str) -> None:
        self.product = product
        self.view = view
        model_dir = CONFIG.paths.patchcore_models_root / product / view
        if not (model_dir / "model.ckpt").exists():
            raise FileNotFoundError(f"PatchCore checkpoint missing: {model_dir / 'model.ckpt'}")

        self._engine = PatchCoreEngine.load(model_dir)

        source = self._camera_source()
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open camera source: {source}")

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._capture.set(cv2.CAP_PROP_FPS, 30)

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="patchcore-inference", daemon=True)
        self._thread.start()

    def _camera_source(self):
        # Keep config handling simple for V1; manual editing is intentional.
        source = 0
        config_path = CONFIG.configs_root / "app.yaml"
        if config_path.exists():
            try:
                import yaml
                config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                source = config.get("camera", {}).get("source", 0)
            except Exception:
                pass
        if isinstance(source, str) and source.isdigit():
            return int(source)
        return source

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._capture is not None:
            self._capture.release()

    def _loop(self) -> None:
        assert self._capture is not None
        assert self._engine is not None

        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if not ok or frame is None:
                time.sleep(0.1)
                continue

            self._frame_id += 1
            start = time.perf_counter()

            try:
                result = self._engine.predict(frame)
                system = self._monitor.snapshot().to_dict()
                loop_ms = (time.perf_counter() - start) * 1000.0

                now = time.perf_counter()
                dt = now - self._last_frame_time
                self._last_frame_time = now
                fps = 1.0 / dt if dt > 0 else 0.0

                status = "ANOMALY" if result.is_anomaly else "NORMAL"

                composite = make_composite(
                    frame,
                    result.heatmap_bgr,
                    result.overlay_bgr,
                    self.product,
                    self.view,
                    result.score,
                    result.threshold,
                    status,
                    result.inference_ms,
                )

                snapshot = RuntimeSnapshot(
                    frame_id=self._frame_id,
                    product=self.product,
                    view=self.view,
                    score=result.score,
                    threshold=result.threshold,
                    status=status,
                    inference_ms=result.inference_ms,
                    fps=fps,
                    system={**system, "pipeline_ms": loop_ms},
                    raw_jpeg=encode_jpeg(frame),
                    heatmap_jpeg=encode_jpeg(result.heatmap_bgr),
                    overlay_jpeg=encode_jpeg(result.overlay_bgr),
                    composite_jpeg=encode_jpeg(composite),
                )

                with self._lock:
                    self._latest = snapshot
            except Exception as exc:
                print(f"Inference error: {exc}")
                time.sleep(0.05)

    def snapshot(self) -> RuntimeSnapshot | None:
        with self._lock:
            return self._latest


def encode_jpeg(image: np.ndarray, quality: int = 80) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return encoded.tobytes()


def make_composite(
    original: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    product: str,
    view: str,
    score: float,
    threshold: float,
    status: str,
    inference_ms: float,
) -> np.ndarray:
    def labeled(image, label):
        canvas = image.copy()
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 38), (20, 20, 20), -1)
        cv2.putText(canvas, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        return canvas

    a = labeled(original, "ORIGINAL")
    b = labeled(heatmap, "HEATMAP")
    c = labeled(overlay, "OVERLAY")

    row = np.concatenate([a, b, c], axis=1)
    header = np.zeros((70, row.shape[1], 3), dtype=np.uint8)
    text = f"{product} | {view} | {status} | score={score:.4f} | threshold={threshold:.4f} | inference={inference_ms:.1f} ms"
    cv2.putText(header, text, (15, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return np.concatenate([header, row], axis=0)