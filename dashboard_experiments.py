"""Experiments tab — queue persistence, multi-step chaining, and the Add/Extend/Submit modals."""

import json
import time
from pathlib import Path

from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Button, DataTable, Footer, Input, Label
from textual.screen import ModalScreen

_TRAINING_DIR = Path.cwd()
QUEUE_FILE    = _TRAINING_DIR / "experiments.json"
_params_file  = _TRAINING_DIR / "default_params.json"


def _load_defaults() -> dict:
    return json.loads(_params_file.read_text()) if _params_file.exists() else {}


def _collect_params(modal: ModalScreen) -> dict:
    """Collect all param fields from a modal, casting to the type of the default."""
    result = {}
    for key, default_val in _load_defaults().items():
        raw = modal.query_one(f"#param-{key}", Input).value.strip()
        val = raw if raw else default_val
        result[key] = int(float(val)) if isinstance(default_val, int) else float(val)
    return result


# ---------------------------------------------------------------------------
# Modals
# ---------------------------------------------------------------------------
class QueueModal(ModalScreen):
    """Add an experiment to the queue, optionally with chained tasks."""
    BINDINGS = [
        Binding("ctrl+s", "confirm", "Add to Queue"),
        Binding("escape", "dismiss", "Cancel"),
    ]

    def __init__(self):
        super().__init__()
        self._num_tasks = 1

    def compose(self):
        yield Footer()
        with Vertical(id="spawn-dialog"):
            yield Label("Add Experiment to Queue", id="spawn-title")
            yield Label("run_name")
            yield Input(placeholder=f"exp_{int(time.time())}", id="run-name")
            with Vertical(id="tasks-container"):
                yield Label("── Task 1", markup=False, id="task-label-0")
                for key, val in _load_defaults().items():
                    yield Label(key)
                    yield Input(value=str(val), id=f"task0-param-{key}")

    def action_confirm(self) -> None:
        defaults = _load_defaults()
        tasks = []
        for t in range(self._num_tasks):
            task_params = {}
            for key, default_val in defaults.items():
                raw = self.query_one(f"#task{t}-param-{key}", Input).value.strip()
                val = raw if raw else default_val
                task_params[key] = int(float(val)) if isinstance(default_val, int) else float(val)
            tasks.append(task_params)
        result = {
            "run_name": self.query_one("#run-name", Input).value or f"exp_{int(time.time())}",
            "tasks": tasks,
        }
        self.dismiss(result)


class AddStepModal(ModalScreen):
    """Add a step to an existing queued experiment."""
    BINDINGS = [
        Binding("ctrl+s", "confirm", "Add Step"),
        Binding("escape", "dismiss", "Cancel"),
    ]

    def compose(self):
        yield Footer()
        with Vertical(id="spawn-dialog"):
            yield Label("Add Step to Experiment", id="spawn-title")
            for key, val in _load_defaults().items():
                yield Label(key)
                yield Input(value=str(val), id=f"param-{key}")

    def action_confirm(self) -> None:
        self.dismiss(_collect_params(self))


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------
class ExperimentsMixin:
    """Mixin for Dashboard providing all Experiments-tab logic.

    Relies on the host class (Dashboard) providing:
        self.experiment_queue, self.selected_queue_idx, self.runs,
        self._free_gpu(), self._gpu_by_index(), self._on_spawn_result(),
        self._checkpoints(), self.notify(), self.push_screen(), self.query_one()
    """

    def _save_queue(self) -> None:
        with open(QUEUE_FILE, "w") as f:
            json.dump(self.experiment_queue, f, indent=2)

    def _load_queue(self) -> None:
        if not QUEUE_FILE.exists():
            return
        with open(QUEUE_FILE) as f:
            self.experiment_queue = json.load(f)

    def _spawn_chain_task(self, run) -> None:
        next_idx = run.chain_task_idx + 1
        exp = next((e for e in self.experiment_queue if e.get("run_name") == run.chain_experiment), None)
        if exp is None:
            return
        tasks = exp.get("tasks") or [exp]
        if next_idx >= len(tasks):
            return
        candidates = self._checkpoints(run.run_name)
        checkpoint = str(max(candidates, key=lambda p: p.stat().st_mtime)) if candidates else None
        gpu = self._gpu_by_index(run.gpu_id) or self._free_gpu()
        if gpu is None:
            self.notify(f"No GPU available for chain step {next_idx + 1}", severity="error")
            return
        launch_config = {
            **tasks[next_idx],
            "run_name":   run.run_name,
            "gpu_id":     gpu.index,
            "checkpoint": checkpoint,
        }
        self.runs.remove(run)
        self._on_spawn_result(launch_config)
        new_run = self.runs[-1]
        new_run.chain_experiment  = run.chain_experiment
        new_run.chain_task_idx    = next_idx
        new_run.chain_total_tasks = run.chain_total_tasks
        self.notify(
            f"{run.run_name}: step {next_idx + 1}/{run.chain_total_tasks} started",
            timeout=5,
        )

    def _refresh_queue_table(self) -> None:
        table = self.query_one("#queue-table", DataTable)
        table.clear()
        defaults = _load_defaults()
        for exp in self.experiment_queue:
            tasks = exp.get("tasks") or [exp]
            task1 = tasks[0]
            changed = ", ".join(
                f"{k}={task1[k]}" for k in defaults
                if k in task1 and task1[k] != defaults[k]
            )
            suffix = f"  [{len(tasks)} steps]" if len(tasks) > 1 else ""
            table.add_row(exp.get("run_name", "-"), (changed or "(defaults)") + suffix)
        if 0 <= self.selected_queue_idx < len(self.experiment_queue):
            table.move_cursor(row=self.selected_queue_idx)

    # -----------------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------------
    def action_add_to_queue(self) -> None:
        self.push_screen(QueueModal(), self._on_queue_result)

    def _on_queue_result(self, config: dict | None) -> None:
        if config is None:
            return
        self.experiment_queue.append(config)
        self._save_queue()
        self._refresh_queue_table()
        self.notify(f"Queued {config['run_name']}", timeout=3)

    def action_add_step(self) -> None:
        if self.selected_queue_idx < 0 or self.selected_queue_idx >= len(self.experiment_queue):
            self.notify("No experiment selected.", severity="warning")
            return
        self.push_screen(AddStepModal(), self._on_add_step_result)

    def _on_add_step_result(self, params: dict | None) -> None:
        if params is None:
            return
        exp = self.experiment_queue[self.selected_queue_idx]
        if "tasks" not in exp:
            flat = {k: v for k, v in exp.items() if k in _load_defaults()}
            exp["tasks"] = [flat]
        exp["tasks"].append(params)
        self._save_queue()
        self._refresh_queue_table()
        self.notify(f"Added step {len(exp['tasks'])} to {exp['run_name']}", timeout=3)

    def action_remove_from_queue(self) -> None:
        if self.selected_queue_idx < 0 or self.selected_queue_idx >= len(self.experiment_queue):
            return
        removed = self.experiment_queue.pop(self.selected_queue_idx)
        self.selected_queue_idx = min(self.selected_queue_idx, len(self.experiment_queue) - 1)
        self._save_queue()
        self._refresh_queue_table()
        self.notify(f"Removed {removed['run_name']} from queue", timeout=3)

    def action_submit_queue(self) -> None:
        self._submit_selected()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "queue-submit":
            self._submit_selected()

    def _submit_selected(self) -> None:
        if self.selected_queue_idx < 0 or self.selected_queue_idx >= len(self.experiment_queue):
            self.notify("No experiment selected.", severity="warning")
            return
        exp = self.experiment_queue[self.selected_queue_idx]
        gpu = self._free_gpu()
        if gpu is None:
            self.notify("No free GPU available.", severity="error")
            return
        tasks = exp.get("tasks") or [exp]
        launch_config = {**tasks[0], "run_name": exp["run_name"], "gpu_id": gpu.index}
        self._on_spawn_result(launch_config)
        if len(tasks) > 1:
            new_run = self.runs[-1]
            new_run.chain_experiment  = exp["run_name"]
            new_run.chain_task_idx    = 0
            new_run.chain_total_tasks = len(tasks)
        self.notify(f"Submitted {exp['run_name']} on GPU {gpu.index}", timeout=5)
