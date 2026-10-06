from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from anomalib.models import Patchcore


@dataclass(frozen=True)
class PatchCoreResult:
    score: float
    threshold: float
    is_anomaly: bool
    anomaly_map: np.ndarray
    heatmap_bgr: np.ndarray
    overlay_bgr: np.ndarray
    inference_ms: float


class PatchCoreEngine:
    """Small runtime wrapper around Anomalib PatchCore.

    The engine loads a trained checkpoint once and exposes a plain NumPy/OpenCV
    interface. FastAPI, a CLI, a video pipeline, or a future ROS2 node can use
    this class without knowing anything about Anomalib internals.
    """

    def __init__(self, model, metadata: dict, device: torch.device):
        self.model = model
        self.metadata = metadata
        self.device = device
        self.threshold = float(metadata["threshold"])
        self.image_size = tuple(metadata.get("image_size", [256, 256]))

    @classmethod
    def load(cls, model_dir: Path, device: str = "auto") -> "PatchCoreEngine":
        model_dir = Path(model_dir)
        metadata = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
        ckpt_path = model_dir / "model.ckpt"

        if device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved = device
        torch_device = torch.device(resolved)

        model = Patchcore(
            backbone=metadata["backbone"],
            layers=tuple(metadata["layers"]),
            pre_trained=False,
            coreset_sampling_ratio=float(metadata["coreset_sampling_ratio"]),
            num_neighbors=int(metadata["num_neighbors"]),
            pre_processor=Patchcore.configure_pre_processor(
                image_size=tuple(metadata.get("image_size", [256, 256]))
            ),
            post_processor=False,
            evaluator=False,
            visualizer=False,
        )

        try:
            checkpoint = torch.load(
                ckpt_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location="cpu")

        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"PatchCore load warning: missing={len(missing)}, unexpected={len(unexpected)}")

        model.to(torch_device)
        model.eval()

        return cls(model=model, metadata=metadata, device=torch_device)

    @torch.inference_mode()
    def predict(self, frame_bgr: np.ndarray) -> PatchCoreResult:
        start = perf_counter()

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0
        tensor = tensor.unsqueeze(0)

        tensor = self.model.pre_processor(tensor)
        tensor = tensor.to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        prediction = self.model.model(tensor)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        score = float(prediction.pred_score.detach().float().cpu().item())
        anomaly_map = prediction.anomaly_map.detach().float().cpu().numpy()[0]
        anomaly_map = np.squeeze(anomaly_map)

        anomaly_map = cv2.resize(
            anomaly_map,
            (frame_bgr.shape[1], frame_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        normalized = cv2.normalize(
            anomaly_map,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        overlay_bgr = cv2.addWeighted(
            frame_bgr,
            0.55,
            heatmap_bgr,
            0.45,
            0,
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (perf_counter() - start) * 1000.0

        return PatchCoreResult(
            score=score,
            threshold=self.threshold,
            is_anomaly=score > self.threshold,
            anomaly_map=anomaly_map,
            heatmap_bgr=heatmap_bgr,
            overlay_bgr=overlay_bgr,
            inference_ms=inference_ms,
        )