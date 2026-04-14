"""TrainingRun dataclass — per-run state and log streaming."""

import queue
import time
from dataclasses import dataclass, field

from ripe_grm.compute_backend_client import JobHandle


# ---------------------------------------------------------------------------
# TrainingRun
# ---------------------------------------------------------------------------
@dataclass
class TrainingRun:
    run_name:   str
    gpu_id:     int | None
    params:     dict = field(default_factory=dict)
    handle:     JobHandle | None = None
    start_time: float = field(default_factory=time.time)
    stopped_at: float | None = None
    log_lines:  list[str] = field(default_factory=list)
    log_queue:  queue.Queue = field(default_factory=queue.Queue)
    status:     str = "running"
    steps:      int = 0
    log_file:   str = ""
    alive_failures:    int = 0
    chain_experiment:  str | None = None
    chain_task_idx:    int = 0
    chain_total_tasks: int = 1

    @property
    def elapsed(self) -> str:
        if self.status == "running":
            end = time.time()
        else:
            end = self.stopped_at if self.stopped_at is not None else self.start_time
        s = int(end - self.start_time)
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{sec:02d}"

    @property
    def device_label(self) -> str:
        if self.gpu_id is None:
            return "CPU"
        if self.handle and hasattr(self.handle, "backend_id"):
            return f"GPU({self.handle.backend_id}, {self.gpu_id})"
        return f"GPU {self.gpu_id}"

    def is_alive(self) -> bool:
        return self.handle.is_running() if self.handle else False

    def actual_run_name(self) -> str | None:
        """Query the backend for what run name the GPU slot is actually tracking.

        Returns the backend's run_name if a job is running on the slot, else None.
        """
        if not self.handle or not hasattr(self.handle, "job_status"):
            return None
        status = self.handle.job_status()
        if status and status.get("running"):
            return status.get("run_name")
        return None

    def terminate(self) -> None:
        if self.handle:
            self.handle.cancel()

    def start_reader(self) -> None:
        if self.handle:
            self.log_queue = self.handle.open_log_reader()

    def drain_queue(self) -> list[str]:
        lines = []
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item is None:
                    break
                lines.append(item)
        except queue.Empty:
            pass
        return lines
