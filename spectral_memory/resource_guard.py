"""Bound one model process and record resource use without changing system settings."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading
import time


GIB = 1024**3


def configure_threads() -> None:
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "2"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def gpu_snapshot() -> dict:
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    values = [int(value.strip()) for value in result.stdout.strip().split(",")]
    if len(values) != 5:
        raise RuntimeError("Unexpected NVIDIA resource query response")
    return dict(zip(("gpu_total_mib", "gpu_used_mib", "gpu_free_mib", "gpu_temperature_c", "gpu_utilization_pct"), values))


class ResourceGuard:
    def __init__(self, output: Path, device: str, timeout_seconds: int = 300):
        if device not in {"cpu", "cuda"}:
            raise ValueError("Only one CPU or cuda:0 process is supported")
        self.output = output
        self.device = device
        self.timeout_seconds = timeout_seconds
        self.started = time.monotonic()
        self.stopped = threading.Event()
        self.phase = "initializing"
        self.samples = []
        self.process = None
        self.thread = None
        self.lock_stream = None
        self.configuration = {
            "device": device, "cpu_threads": 2, "interop_threads": 1,
            "cuda_allocator_fraction": 0.65 if device == "cuda" else None,
            "minimum_initial_available_ram_gib": 12, "minimum_running_available_ram_gib": 6,
            "maximum_process_rss_gib": 8, "maximum_gpu_temperature_c": 78,
            "maximum_initial_gpu_temperature_c": 65, "timeout_seconds": timeout_seconds,
            "maximum_initial_gpu_used_mib": 256, "minimum_initial_gpu_free_mib": 6500,
            "monitor_interval_seconds": 2,
            "limitation": "Process and allocator limits cannot prevent hypervisor, driver, firmware or hardware faults.",
        }

    def _sample(self) -> dict:
        import psutil

        sample = {"elapsed_seconds": time.monotonic() - self.started, "phase": self.phase,
                  "process_rss_bytes": self.process.memory_info().rss,
                  "available_ram_bytes": psutil.virtual_memory().available}
        if self.device == "cuda":
            sample.update(gpu_snapshot())
        return sample

    def _record(self, sample: dict) -> None:
        self.samples.append(sample)
        with (self.output / "resources.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(sample) + "\n")

    def _failure(self, sample: dict) -> str | None:
        if sample["elapsed_seconds"] > self.timeout_seconds:
            return "run_timeout"
        if sample["available_ram_bytes"] < 6 * GIB:
            return "low_system_memory"
        if sample["process_rss_bytes"] > 8 * GIB:
            return "process_memory_limit"
        if self.device == "cuda" and sample["gpu_temperature_c"] >= 78:
            return "gpu_temperature_limit"
        if self.device == "cuda" and sample["gpu_free_mib"] < 1024:
            return "low_free_gpu_memory"
        return None

    def _monitor(self) -> None:
        while not self.stopped.wait(2):
            try:
                sample = self._sample()
                self._record(sample)
                reason = self._failure(sample)
            except Exception as error:
                reason = f"resource_monitor_failed: {type(error).__name__}: {error}"
            if self.stopped.is_set():
                return
            if reason:
                try:
                    (self.output / "resource_stop.json").write_text(
                        json.dumps({"status": "stopped", "reason": reason, "phase": self.phase}, indent=2),
                        encoding="utf-8",
                    )
                finally:
                    os._exit(73)

    def __enter__(self):
        import psutil

        configure_threads()
        if self.device == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0"):
            raise RuntimeError("The GPU guard requires physical cuda:0 without CUDA_VISIBLE_DEVICES remapping")
        self.process = psutil.Process()
        affinity = self.process.cpu_affinity()
        self.process.cpu_affinity(affinity[:2])
        if os.name == "nt":
            self.process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        self.configuration["cpu_affinity"] = self.process.cpu_affinity()
        lock_path = Path(__file__).resolve().parents[1] / ".cache" / "limited_model.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_stream = lock_path.open("a+b")
        if lock_path.stat().st_size == 0:
            self.lock_stream.write(b"0")
            self.lock_stream.flush()
        self.lock_stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock_stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            sample = self._sample()
            if sample["available_ram_bytes"] < 12 * GIB:
                raise RuntimeError("At least 12 GiB available RAM is required before loading")
            if self.device == "cuda" and (
                sample["gpu_used_mib"] > 256 or sample["gpu_free_mib"] < 6500
                or sample["gpu_temperature_c"] > 65
            ):
                raise RuntimeError("GPU preflight failed: device must be idle, cool and have 6500 MiB free")
            self._record(sample)
            (self.output / "resource_limits.json").write_text(json.dumps(self.configuration, indent=2), encoding="utf-8")
            self.thread = threading.Thread(target=self._monitor, daemon=True)
            self.thread.start()
            return self
        except BaseException:
            self.lock_stream.close()
            raise

    def configure_torch(self) -> None:
        import torch

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            torch.cuda.set_per_process_memory_fraction(0.65, 0)
            torch.cuda.reset_peak_memory_stats(0)

    def __exit__(self, exception_type, exception, traceback):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=7)
        if self.lock_stream is not None:
            self.lock_stream.close()
        (self.output / "resource_summary.json").write_text(json.dumps({
            "status": "complete" if exception is None else "failed", "samples": len(self.samples),
            "elapsed_seconds": time.monotonic() - self.started,
            "peak_sampled_rss_bytes": max((sample["process_rss_bytes"] for sample in self.samples), default=0),
            "minimum_sampled_available_ram_bytes": min((sample["available_ram_bytes"] for sample in self.samples), default=0),
            "peak_sampled_gpu_temperature_c": max((sample.get("gpu_temperature_c", 0) for sample in self.samples), default=0),
            "sampling_limitation": "Two-second telemetry can miss short peaks; CUDA allocator peaks are recorded separately.",
            "exception": str(exception) if exception is not None else None,
        }, indent=2), encoding="utf-8")
