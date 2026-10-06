# Harting PatchCore V1 vertical prototype

This patch adds the minimum end-to-end prototype:

1. Prepare one product/view dataset.
2. Train one Anomalib PatchCore model.
3. Calibrate a normal-only threshold from validation images.
4. Load the checkpoint once in a reusable `PatchCoreEngine`.
5. Read a camera continuously.
6. Produce score + heatmap + overlay.
7. Serve a browser dashboard with inference and GPU/CPU/RAM telemetry.

## Files

- `configs/app.yaml` — manual product/view and camera selection.
- `scripts/prepare_demo_dataset.py` — raw -> train/val/test for one product/view.
- `scripts/train_patchcore.py` — Anomalib PatchCore training and threshold calibration.
- `src/harting_anomaly/models/patchcore_engine.py` — reusable runtime engine.
- `src/harting_anomaly/inference/runtime.py` — camera + inference worker.
- `src/harting_anomaly/monitoring/system_monitor.py` — CPU/RAM/NVIDIA telemetry.
- `src/harting_anomaly/app.py` — FastAPI + MJPEG + WebSocket.
- `frontend/index.html` — dedicated dashboard.

## Important prototype limitation

The dataset splitter currently splits individual images. For production, replace this with piece/session grouped splitting before reporting model performance.