# from __future__ import annotations

# import json
# from dataclasses import dataclass
# from pathlib import Path
# from time import perf_counter

# import cv2
# import numpy as np
# import torch
# from anomalib.models import Patchcore


# @dataclass(frozen=True)
# class PatchCoreResult:
#     score: float
#     threshold: float
#     is_anomaly: bool
#     anomaly_map: np.ndarray
#     heatmap_bgr: np.ndarray
#     overlay_bgr: np.ndarray
#     inference_ms: float


# class PatchCoreEngine:
#     """Small runtime wrapper around Anomalib PatchCore.

#     The engine loads a trained checkpoint once and exposes a plain NumPy/OpenCV
#     interface. FastAPI, a CLI, a video pipeline, or a future ROS2 node can use
#     this class without knowing anything about Anomalib internals.
#     """

#     def __init__(self, model, metadata: dict, device: torch.device):
#         self.model = model
#         self.metadata = metadata
#         self.device = device
#         self.threshold = float(metadata["threshold"])
#         self.image_size = tuple(metadata.get("image_size", [256, 256]))

#     @classmethod
#     def load(cls, model_dir: Path, device: str = "auto") -> "PatchCoreEngine":
#         model_dir = Path(model_dir)
#         metadata = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
#         ckpt_path = model_dir / "model.ckpt"

#         if device == "auto":
#             resolved = "cuda" if torch.cuda.is_available() else "cpu"
#         else:
#             resolved = device
#         torch_device = torch.device(resolved)

#         model = Patchcore(
#             backbone=metadata["backbone"],
#             layers=tuple(metadata["layers"]),
#             pre_trained=False,
#             coreset_sampling_ratio=float(metadata["coreset_sampling_ratio"]),
#             num_neighbors=int(metadata["num_neighbors"]),
#             pre_processor=Patchcore.configure_pre_processor(
#                 image_size=tuple(metadata.get("image_size", [256, 256]))
#             ),
#             post_processor=False,
#             evaluator=False,
#             visualizer=False,
#         )

#         try:
#             checkpoint = torch.load(
#                 ckpt_path,
#                 map_location="cpu",
#                 weights_only=False,
#             )
#         except TypeError:
#             checkpoint = torch.load(ckpt_path, map_location="cpu")

#         state_dict = checkpoint.get("state_dict", checkpoint)
#         missing, unexpected = model.load_state_dict(state_dict, strict=False)
#         if missing or unexpected:
#             print(f"PatchCore load warning: missing={len(missing)}, unexpected={len(unexpected)}")

#         model.to(torch_device)
#         model.eval()

#         return cls(model=model, metadata=metadata, device=torch_device)

#     @torch.inference_mode()
#     def predict(self, frame_bgr: np.ndarray) -> PatchCoreResult:
#         start = perf_counter()

#         rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#         tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0
#         tensor = tensor.unsqueeze(0)

#         tensor = self.model.pre_processor(tensor)
#         tensor = tensor.to(self.device)

#         if self.device.type == "cuda":
#             torch.cuda.synchronize(self.device)

#         prediction = self.model.model(tensor)

#         if self.device.type == "cuda":
#             torch.cuda.synchronize(self.device)

#         score = float(prediction.pred_score.detach().float().cpu().item())
#         anomaly_map = prediction.anomaly_map.detach().float().cpu().numpy()[0]
#         anomaly_map = np.squeeze(anomaly_map)

#         anomaly_map = cv2.resize(
#             anomaly_map,
#             (frame_bgr.shape[1], frame_bgr.shape[0]),
#             interpolation=cv2.INTER_LINEAR,
#         )

#         normalized = cv2.normalize(
#             anomaly_map,
#             None,
#             0,
#             255,
#             cv2.NORM_MINMAX,
#         ).astype(np.uint8)
#         heatmap_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

#         overlay_bgr = cv2.addWeighted(
#             frame_bgr,
#             0.55,
#             heatmap_bgr,
#             0.45,
#             0,
#         )

#         if self.device.type == "cuda":
#             torch.cuda.synchronize(self.device)
#         inference_ms = (perf_counter() - start) * 1000.0

#         return PatchCoreResult(
#             score=score,
#             threshold=self.threshold,
#             is_anomaly=score > self.threshold,
#             anomaly_map=anomaly_map,
#             heatmap_bgr=heatmap_bgr,
#             overlay_bgr=overlay_bgr,
#             inference_ms=inference_ms,
#         )

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
    """Production/runtime wrapper for one trained Anomalib PatchCore model.

    Important:
        The input preprocessing is performed exactly once here.
        Raw camera frames are BGR uint8. They are converted to RGB [0, 1],
        moved to the runtime device, then passed through Anomalib's PatchCore
        preprocessor (resize + ImageNet normalization).

        We intentionally call ``self.model.model`` after preprocessing so the
        Lightning-level preprocessor is not applied a second time.
    """

    def __init__(self, model: Patchcore, metadata: dict, device: torch.device):
        self.model = model
        self.metadata = metadata
        self.device = device
        self.threshold = float(metadata["threshold"])
        self.image_size = tuple(metadata.get("image_size", [256, 256]))

    @classmethod
    def load(cls, model_dir: Path, device: str = "auto") -> "PatchCoreEngine":
        model_dir = Path(model_dir)
        metadata_path = model_dir / "model.json"
        checkpoint_path = model_dir / "model.ckpt"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata: {metadata_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        if device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved = device

        torch_device = torch.device(resolved)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

        pre_processor = Patchcore.configure_pre_processor(
            image_size=tuple(metadata.get("image_size", [256, 256]))
        )

        model = Patchcore(
            backbone=metadata["backbone"],
            layers=tuple(metadata["layers"]),
            pre_trained=False,
            coreset_sampling_ratio=float(metadata["coreset_sampling_ratio"]),
            num_neighbors=int(metadata["num_neighbors"]),
            pre_processor=pre_processor,
            post_processor=False,
            evaluator=False,
            visualizer=False,
        )

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError("Checkpoint does not contain a valid state_dict")

        # Do not silently accept an incompatible checkpoint. A PatchCore model
        # is small enough here that a strict state-dict check is preferable.
        model.load_state_dict(state_dict, strict=True)

        model.to(torch_device)
        model.eval()

        return cls(model=model, metadata=metadata, device=torch_device)

    def _prepare_input(self, frame_bgr: np.ndarray) -> torch.Tensor:
        if frame_bgr is None:
            raise ValueError("frame_bgr is None")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(
                f"Expected BGR image with shape HxWx3, got {frame_bgr.shape}"
            )

        # Same semantic input expected by Anomalib's read_image(): RGB, float32,
        # scaled to [0, 1]. Camera input comes from OpenCV as BGR uint8.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        tensor = tensor.unsqueeze(0).float().div(255.0)

        # Move BEFORE preprocessing. Anomalib's Lightning PreProcessor callback
        # normally sees a batch already moved to the target device. Keeping the
        # runtime wrapper in the same order removes unnecessary CPU/GPU path
        # differences in Resize/Normalize.
        tensor = tensor.to(self.device, non_blocking=self.device.type == "cuda")
        tensor = self.model.pre_processor(tensor)
        return tensor

    @torch.inference_mode()
    def predict(self, frame_bgr: np.ndarray) -> PatchCoreResult:
        start = perf_counter()

        tensor = self._prepare_input(frame_bgr)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        # IMPORTANT: preprocessing has already happened exactly once above.
        prediction = self.model.model(tensor)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        score = float(
            prediction.pred_score.detach().float().cpu().reshape(-1)[0].item()
        )

        anomaly_map = prediction.anomaly_map.detach().float().cpu().numpy()[0]
        anomaly_map = np.squeeze(anomaly_map)
        anomaly_map = cv2.resize(
            anomaly_map,
            (frame_bgr.shape[1], frame_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        # Display-only visualization. This is intentionally NOT used for
        # thresholding. The colormap is min-max normalized per frame.
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