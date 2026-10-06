from __future__ import annotations

import platform
import time
from dataclasses import asdict, dataclass

import psutil

try:
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None


@dataclass
class SystemTelemetry:
    timestamp: float
    cpu_name: str
    cpu_percent: float
    ram_percent: float
    ram_used_mb: float
    gpu_name: str | None
    gpu_percent: float | None
    gpu_memory_used_mb: float | None
    gpu_memory_total_mb: float | None
    gpu_temperature_c: float | None

    def to_dict(self) -> dict:
        return asdict(self)


class SystemMonitor:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self._last_snapshot = None
        self._last_snapshot_time = 0.0
        self._nvml_ready = False
        self._handle = None
        self._init_nvml()

    def _init_nvml(self) -> None:
        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() > 0:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml_ready = True
        except Exception:
            self._nvml_ready = False

    def snapshot(self) -> SystemTelemetry:
        now = time.time()
        if self._last_snapshot is not None and (now - self._last_snapshot_time) < self.interval_seconds:
            return self._last_snapshot

        memory = psutil.virtual_memory()

        gpu_name = None
        gpu_percent = None
        gpu_memory_used_mb = None
        gpu_memory_total_mb = None
        gpu_temperature_c = None

        if self._nvml_ready and self._handle is not None:
            try:
                gpu_name = pynvml.nvmlDeviceGetName(self._handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode(errors="replace")
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                gpu_percent = float(utilization.gpu)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                gpu_memory_used_mb = mem.used / (1024 * 1024)
                gpu_memory_total_mb = mem.total / (1024 * 1024)
                try:
                    gpu_temperature_c = float(
                        pynvml.nvmlDeviceGetTemperature(
                            self._handle,
                            pynvml.NVML_TEMPERATURE_GPU,
                        )
                    )
                except Exception:
                    pass
            except Exception:
                pass

        snapshot = SystemTelemetry(
            timestamp=now,
            cpu_name=platform.processor() or platform.machine(),
            cpu_percent=float(psutil.cpu_percent(interval=None)),
            ram_percent=float(memory.percent),
            ram_used_mb=memory.used / (1024 * 1024),
            gpu_name=gpu_name,
            gpu_percent=gpu_percent,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
            gpu_temperature_c=gpu_temperature_c,
        )
        self._last_snapshot = snapshot
        self._last_snapshot_time = now
        return snapshot