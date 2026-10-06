from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, StreamingResponse

from harting_anomaly.core.config import CONFIG
from harting_anomaly.inference.runtime import InferenceRuntime

runtime = InferenceRuntime()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    import yaml
    config_path = CONFIG.configs_root / "app.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    selection = config.get("selection", {})
    product = selection.get("product")
    view = selection.get("view")
    if not product or not view:
        raise RuntimeError("configs/app.yaml must define selection.product and selection.view")
    runtime.start(product, view)
    try:
        yield
    finally:
        runtime.stop()


app = FastAPI(title="Harting PatchCore V1", lifespan=lifespan)


@app.get("/")
def index():
    path = CONFIG.paths.frontend_root / "index.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(path)


def mjpeg_generator(kind: str):
    while True:
        snapshot = runtime.snapshot()
        if snapshot is None:
            import time
            time.sleep(0.05)
            continue

        if kind == "raw":
            payload = snapshot.raw_jpeg
        elif kind == "heatmap":
            payload = snapshot.heatmap_jpeg
        elif kind == "overlay":
            payload = snapshot.overlay_jpeg
        else:
            payload = snapshot.composite_jpeg

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n"
            + payload + b"\r\n"
        )
        import time
        time.sleep(0.02)


@app.get("/stream/{kind}.mjpg")
def stream(kind: str):
    if kind not in {"raw", "heatmap", "overlay", "composite"}:
        raise HTTPException(status_code=404, detail="Unknown stream")
    return StreamingResponse(
        mjpeg_generator(kind),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws/metrics")
async def metrics(websocket: WebSocket):
    await websocket.accept()
    while True:
        snapshot = runtime.snapshot()
        if snapshot is not None:
            await websocket.send_json(snapshot.public_dict())
        await asyncio.sleep(0.25)