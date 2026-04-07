"""TrainingRun dataclass and SpawnModal — per-run state, log streaming, and the spawn dialog."""

import queue
import time
from dataclasses import dataclass, field

from ripe_autotrain.compute_backend_client import JobHandle
from ripe_autotrain.dashboard_experiments import _load_defaults, _collect_params
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label


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
    steps_offset: int = 0
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


# ---------------------------------------------------------------------------
# SpawnModal
# ---------------------------------------------------------------------------
class SpawnModal(ModalScreen):
    BINDINGS = [("escape", "dismiss", "Cancel")]

    def __init__(self, next_gpu: int | None, n_gpus: int):
        super().__init__()
        self._next_gpu = next_gpu
        self._n_gpus   = n_gpus

    def compose(self):
        default_device = str(self._next_gpu) if self._next_gpu is not None else "0"
        with Vertical(id="spawn-dialog"):
            yield Label("Spawn MJX Training Run", id="spawn-title")
            yield Label("run_name")
            yield Input(placeholder=f"run_{int(time.time())}", id="run-name")
            yield Label(f"gpu_id  (0–{self._n_gpus - 1})")
            yield Input(value=default_device, id="gpu-id")
            with VerticalScroll(id="spawn-params"):
                for key, val in _load_defaults().items():
                    yield Label(key)
                    yield Input(value=str(val), id=f"param-{key}")
            with Horizontal(id="spawn-buttons"):
                yield Button("Spawn", variant="success", id="spawn-confirm")
                yield Button("Cancel", variant="error", id="spawn-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "spawn-cancel":
            self.dismiss(None)
            return
        try:
            gpu_id = max(0, min(int(self.query_one("#gpu-id", Input).value), self._n_gpus - 1))
        except ValueError:
            gpu_id = 0
        result = _collect_params(self)
        result["run_name"] = self.query_one("#run-name", Input).value or f"run_{int(time.time())}"
        result["gpu_id"]   = gpu_id
        self.dismiss(result)
