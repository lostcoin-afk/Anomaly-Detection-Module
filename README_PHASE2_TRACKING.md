# Anomaly Detection Module — Phase 1 → Phase 2

## Current Status

**Phase 1 is complete and frozen as the baseline.**

Phase 1 currently provides:

- Camera/video input with local-video fallback
- PatchCore anomaly inference
- Anomaly score + threshold classification
- Heatmap and composite visualization
- Raw/composite MJPEG streaming
- FastAPI application
- WebSocket metrics
- Basic application/service structure
- PatchCore diagnostic tooling
- Versioned threshold calibration tooling

Before substantial refactoring, preserve Phase 1 in GitHub as the reference implementation.

Recommended tag:

```text
phase-1-baseline
```

Recommended development branch:

```text
develop/phase-2
```

---

# Phase 1 Baseline

The Phase 1 objective was:

> **Prove that anomaly detection works end-to-end.**

That goal has been achieved.

The Phase 1 implementation should remain reproducible so that Phase 2 changes can always be compared against the known-working baseline.

---

# Important Phase 1 Findings

## PatchCore inference path

The PatchCore runtime score path has been validated.

The runtime wrapper, independent RGB/preprocessing reference, and Anomalib-equivalent inference path now agree to numerical precision.

Therefore:

**Do not modify the PatchCore preprocessing/inference path during Phase 2 without a specific reason and a diagnostic comparison.**

The previously observed large score discrepancy caused by duplicate preprocessing has been resolved.

## Threshold calibration

The current stored threshold belongs to an older score distribution and should not be trusted as the current production threshold.

The correct lifecycle is:

```text
Checkpoint
   ↓
Runtime score path
   ↓
Validation dataset
   ↓
Calibration
   ↓
Versioned calibration record
   ↓
Active threshold
```

Calibration should be an explicit, auditable operation.

Do **not** recalibrate every application startup.

Recommended structure:

```text
models/patchcore/<product>/<view>/
├── model.ckpt
├── model.json
└── calibration/
    ├── calibration_<timestamp>.json
    └── latest.json
```

Recalibration is appropriate after changes such as:

- new checkpoint
- changed training data
- changed preprocessing
- changed image resolution
- changed ROI
- changed camera/lens
- changed lighting/exposure
- significant operating-condition drift
- intentionally changing the operating point

---

# Known Issues Entering Phase 2

## 1. Performance

Current flow is effectively:

```text
High-FPS input
     ↓
Inference
     ↓
Heatmap generation
     ↓
Streaming
     ↓
Frontend
```

The input can produce frames faster than inference and visualization can consume them.

Symptoms:

- frontend feels slow
- frames can become stale
- inference and streaming are coupled
- heatmap/overlay work is performed in the live path
- camera acquisition is affected by downstream processing

The first Phase 2 action should be **profiling**, not immediately adding optimizations.

Measure separately:

```text
capture/acquisition
frame conversion
preprocessing
model inference
anomaly-map generation
heatmap generation
overlay/composite generation
JPEG encoding
streaming
frontend update rate
```

---

## 2. WebSocket shutdown exception

During normal shutdown, the browser/client can disconnect while the server is still executing:

```python
await websocket.send_json(...)
```

This currently produces:

```text
ClientDisconnected
WebSocketDisconnect
```

in the server traceback.

This is a lifecycle-handling problem rather than an anomaly-model failure.

Phase 2 should treat client disconnect as a normal event:

```text
send metrics
    ↓
client disconnects
    ↓
catch disconnect
    ↓
stop send loop
    ↓
cleanup
    ↓
no noisy traceback
```

---

## 3. Camera/Input abstraction

Current input handling is too closely coupled to the existing OpenCV/video implementation.

The module should eventually support:

```text
Camera
Video file
RTSP/network stream
Future industrial sources
```

without changing the inference pipeline.

---

# Phase 2 Goal

Phase 2 is **not only a performance improvement pass**.

The goal is to turn the prototype into a:

> **Modular, performant, extensible Anomaly Detection Module that can plug into the existing Auto Training Platform.**

Target architecture:

```text
Input Source
      ↓
Frame Acquisition
      ↓
Frame Buffer / Scheduler
      ↓
Inference Engine
      ↓
Anomaly Model
      ↓
Post Processing
      ↓
AnomalyResult
   ┌──┼───────────────┐
   ↓  ↓               ↓
Viz Stream          Metrics
```

Core separation principle:

```text
Input should not know about models.
Models should not know about the frontend.
Inference should not depend on visualization.
Visualization should not control acquisition.
Streaming should not control inference.
```

---

# Phase 2 Target Architecture

```text
                         ┌─────────────────────┐
                         │    Input Sources    │
                         │---------------------│
                         │ Camera              │
                         │ Video File          │
                         │ RTSP                │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Frame Acquisition   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Frame Buffer /      │
                         │ Scheduler           │
                         │                     │
                         │ latest-frame /      │
                         │ bounded buffering   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Inference Engine    │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │ Common Anomaly      │
                         │ Model Interface     │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    │               │                │
                    ▼               ▼                ▼
               PatchCore        FastFlow        EfficientAD
                    │               │                │
                    └───────────────┼────────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Common              │
                         │ AnomalyResult       │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    │               │                │
                    ▼               ▼                ▼
              Visualization     Streaming        Metrics
              / Heatmap          / Web API        / WS
```

---

# Phase 2 Priorities

## P0 — Preserve the baseline

- Keep Phase 1 frozen/tagged.
- Create the Phase 2 development branch.
- Record environment/model versions required to reproduce Phase 1.
- Do not refactor `main` directly.

## P1 — Profile first

Build a profile of:

```text
capture FPS
processed FPS
preprocessing
inference
anomaly-map generation
heatmap
overlay
JPEG encoding
streaming
end-to-end latency
GPU utilization
GPU memory
CPU utilization
```

The goal is to determine the actual bottleneck.

## P2 — Decouple acquisition from inference

Introduce controlled frame processing:

```text
Camera
  ↓
continuous acquisition
  ↓
latest/bounded frame
  ↓
inference worker
```

For live inspection, avoid an unlimited queue that causes latency to grow.

Preferred behavior:

> process the newest useful frame rather than accumulating stale frames.

## P3 — Separate inference from visualization/streaming

Inference should produce a result.

Visualization and streaming should consume that result independently.

Potential operating rates:

```text
Input             high FPS
Inference         controlled FPS
Visualization     controlled FPS
Metrics           low FPS
```

Exact rates should be determined from profiling.

## P4 — Fix WebSocket lifecycle handling

Normal client disconnects should terminate the metrics send loop cleanly without an exception traceback.

## P5 — Introduce `FrameSource`

Conceptual interface:

```python
class FrameSource(Protocol):
    def open(self) -> None:
        ...
    def read(self) -> Frame | None:
        ...
    def close(self) -> None:
        ...
```

Initial implementations:

```text
CameraSource
VideoFileSource
RTSPSource
```

The inference system should not depend directly on `cv2.VideoCapture`.

## P6 — Introduce a common anomaly-model interface

Conceptually:

```python
class AnomalyModel(Protocol):
    def predict(self, frame: np.ndarray) -> AnomalyResult:
        ...
```

Initial model:

```text
PatchCoreModel
```

Future models:

```text
FastFlowModel
EfficientADModel
```

The rest of the system should consume `AnomalyResult`, not Anomalib-specific prediction objects.

## P7 — Define common `AnomalyResult`

Conceptually:

```python
@dataclass
class AnomalyResult:
    score: float
    threshold: float
    is_anomaly: bool
    anomaly_map: np.ndarray | None
    inference_ms: float
    model_name: str
    model_version: str
```

The final contract should be aligned with the Auto Training Platform before it becomes a stable public interface.

## P8 — Add FastFlow and EfficientAD

Add these only after the common model contract is stable.

Goal:

```text
same input
   ↓
same module pipeline
   ↓
different anomaly model
   ↓
same result contract
```

## P9 — Benchmark models

Benchmark under controlled conditions:

```text
model
input resolution
device
preprocessing time
inference time
postprocessing time
total latency
throughput
GPU memory
CPU utilization
false positives
false negatives
anomaly quality
```

Model choice should consider both detection quality and industrial runtime requirements.

## P10 — Configuration/model selection

Move toward configuration such as:

```yaml
input:
  type: camera

model:
  type: patchcore
  product: ...
  view: ...

runtime:
  inference_fps: ...
  visualization_fps: ...
  metrics_fps: ...

streaming:
  raw: true
  composite: true
```

Model selection should become configuration-driven rather than hardcoded.

## P11 — Align with Auto Training Platform

Before finalizing interfaces, review the Auto Training System Design and existing entry points/contracts.

The anomaly module should eventually consume platform-managed artifacts/models rather than creating a parallel model-management system.

Expected integration direction:

```text
Auto Training Platform
        │
        ├── Dataset / Artifact
        ├── Model
        ├── Evaluation
        └── Deployment
                 │
                 ▼
        Anomaly Detection Module
                 │
                 ├── load model
                 ├── select model/product/view
                 ├── accept frame
                 └── return AnomalyResult
```

Model lifecycle, artifact identity, provenance, evaluation metadata and deployment/version concepts should align with the platform.

---

# Suggested Internal Package Direction

Refactor gradually rather than creating the whole final architecture at once.

Target direction:

```text
src/harting_anomaly/
├── app.py
├── core/
│   ├── config.py
│   ├── types.py
│   └── lifecycle.py
├── input/
│   ├── base.py
│   ├── camera.py
│   ├── video.py
│   └── rtsp.py
├── pipeline/
│   ├── frame_buffer.py
│   ├── scheduler.py
│   └── worker.py
├── models/
│   ├── base.py
│   ├── patchcore.py
│   ├── fastflow.py
│   └── efficientad.py
├── inference/
│   ├── engine.py
│   └── result.py
├── postprocessing/
│   ├── threshold.py
│   └── heatmap.py
├── streaming/
│   ├── mjpeg.py
│   └── websocket.py
├── monitoring/
│   └── metrics.py
└── api/
    ├── routes.py
    └── websocket.py
```

This is the intended direction, not a mandate to create every file immediately.

---

# Phase 2 Implementation Rules

1. Prefer small, testable commits.
2. Keep the system runnable after each architectural change.
3. Profile before optimizing.
4. Do not change PatchCore preprocessing/inference without diagnostic validation.
5. Do not couple visualization to inference execution.
6. Do not allow unbounded live-frame queues.
7. Do not finalize public interfaces until Auto Training Platform contracts have been reviewed.
8. Keep calibration history immutable/auditable.
9. Benchmark models under identical conditions.

Preferred implementation sequence:

```text
Profile
  ↓
WebSocket lifecycle fix
  ↓
FrameSource abstraction
  ↓
Bounded/latest-frame buffer
  ↓
Inference worker separation
  ↓
Visualization/stream separation
  ↓
AnomalyResult
  ↓
AnomalyModel interface
  ↓
PatchCore adapter
  ↓
Benchmark framework
  ↓
FastFlow
  ↓
EfficientAD
  ↓
Auto Training Platform integration
```

---

# Testing Strategy

## Functional

Verify:

```text
application starts
camera works when available
video fallback works
model loads
inference returns result
raw stream works
composite stream works
WebSocket connects
WebSocket disconnects cleanly
```

## Model correctness

Verify:

```text
same image → deterministic score
runtime score ≈ independent reference score
checkpoint loads cleanly
threshold matches current score space
calibration is reproducible
```

## Performance

Record:

```text
capture FPS
processed FPS
inference FPS
end-to-end latency
buffer depth
GPU utilization
GPU memory
CPU utilization
```

Store benchmark results so Phase 2 improvements are measurable rather than subjective.

---

# Definition of Done — Phase 2

Phase 2 is complete when:

- input sources are abstracted;
- acquisition is decoupled from inference;
- frame buffering is bounded/latest-frame based;
- inference is separated from visualization and streaming;
- normal WebSocket disconnects produce no noisy traceback;
- a common anomaly-model interface exists;
- PatchCore implements it;
- FastFlow implements it;
- EfficientAD implements it;
- models return a common `AnomalyResult`;
- model selection is configuration-driven;
- benchmarking is reproducible;
- performance bottlenecks are measured and documented;
- threshold calibration is versioned and auditable;
- module boundaries are compatible with the Auto Training Platform;
- Phase 2 remains runnable independently of the Auto Training Platform.

---

# Immediate Next Actions

```text
[ ] Confirm Phase 1 commit/tag in GitHub
[ ] Create develop/phase-2
[ ] Freeze current Phase 1 model/checkpoint references
[ ] Run performance profiling
[ ] Fix WebSocket disconnect handling
[ ] Introduce FrameSource abstraction
[ ] Add bounded/latest-frame buffer
[ ] Separate inference worker
[ ] Separate visualization/streaming
[ ] Define AnomalyResult
[ ] Define AnomalyModel
[ ] Adapt PatchCore to the interface
[ ] Build benchmark tooling
[ ] Add FastFlow
[ ] Add EfficientAD
[ ] Review Auto Training Platform interfaces
[ ] Integrate module contracts
```

---

# Change Log

## Phase 1 — Complete

- Initial FastAPI application created.
- Camera/video input created.
- Local-video fallback implemented.
- PatchCore inference implemented.
- Heatmap/composite visualization implemented.
- Raw/composite MJPEG streaming implemented.
- WebSocket metrics implemented.
- PatchCore diagnostic tooling implemented.
- PatchCore runtime/reference score discrepancy investigated and resolved.
- Threshold calibration tooling created.
- Phase 1 baseline prepared for GitHub preservation.

## Phase 2 — In progress

### Not yet implemented

- Performance profiling framework
- WebSocket lifecycle cleanup
- Input abstraction
- Bounded/latest-frame processing
- Inference/visualization/streaming separation
- Common anomaly-model interface
- FastFlow integration
- EfficientAD integration
- Model benchmark framework
- Auto Training Platform integration

---

# Decision Log

## D-001 — Phase 1 is frozen

**Decision:** Preserve Phase 1 before large refactoring.

**Reason:** Maintain a known-working reference.

## D-002 — Latest-frame/bounded buffering

**Decision:** Live inference must not accumulate an unlimited backlog.

**Reason:** Unlimited queues increase latency and produce stale results.

## D-003 — Model abstraction

**Decision:** The application depends on a common anomaly-model interface.

**Reason:** PatchCore, FastFlow, EfficientAD and future models must be interchangeable.

## D-004 — Common result contract

**Decision:** Models return a common `AnomalyResult`.

**Reason:** API, visualization, streaming and metrics should not understand model-specific output structures.

## D-005 — Versioned calibration

**Decision:** Threshold calibration is an explicit versioned operation.

**Reason:** Threshold selection affects production classification and must be reproducible/auditable.

## D-006 — Profile before optimization

**Decision:** Phase 2 performance work begins with measurement.

**Reason:** Prevent premature optimization and identify the actual bottleneck.

---

# Overall Direction

```text
Phase 1
"Does anomaly detection work?"
          ↓
Phase 2
"Can it become a clean, fast, reusable module?"
          ↓
Auto Training Platform
"Can the module become a managed component of the broader ML system?"
          ↓
Production
"Can it reliably operate in an industrial inspection environment?"
```

The Phase 1 implementation is the reference point.

Phase 2 should improve architecture, performance and extensibility while preserving the ability to reproduce and validate the Phase 1 baseline.
