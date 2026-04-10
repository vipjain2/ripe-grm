"""Experiments tab — queue persistence, multi-step chaining, and the Add/Extend/Submit modals."""

import json
import time
from pathlib import Path

from ripe_autotrain.dashboard_log import log_error
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, DataTable, Footer, Input, Label, Select
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

    def __init__(self, backend_ids: list[str]):
        super().__init__()
        self._num_tasks   = 1
        self._backend_ids = backend_ids

    def compose(self):
        gpu_options = [("any", "any")] + [(b, b) for b in self._backend_ids]
        yield Footer()
        with Vertical(id="spawn-dialog"):
            yield Label("Add Experiment to Queue", id="spawn-title")
            yield Label("run_name")
            yield Input(placeholder=f"exp_{int(time.time())}", id="run-name")
            yield Label("gpu")
            yield Select(gpu_options, value="any", allow_blank=False, id="gpu-pref")
            with Vertical(id="tasks-container"):
                yield Label("── Task 1", markup=False, id="task-label-0")
                for key, val in _load_defaults().items():
                    yield Label(key)
                    yield Input(value=str(val), id=f"task0-param-{key}")
            with Horizontal(id="spawn-buttons"):
                yield Button("Add to Queue", variant="success", id="queue-confirm")
                yield Button("Cancel", variant="error", id="queue-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "queue-cancel":
            self.dismiss(None)
        elif event.button.id == "queue-confirm":
            self.action_confirm()

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
        gpu_pref = self.query_one("#gpu-pref", Select).value
        result = {
            "run_name":        self.query_one("#run-name", Input).value or f"exp_{int(time.time())}",
            "gpu_preference":  str(gpu_pref),
            "tasks":           tasks,
        }
        self.dismiss(result)


class AddStepModal(ModalScreen):
    """Add or edit a step in a queued experiment."""
    BINDINGS = [
        Binding("ctrl+s", "confirm", "Save"),
        Binding("escape", "dismiss", "Cancel"),
    ]

    def __init__(self, backend_ids: list[str], initial_params: dict | None = None,
                 initial_gpu_pref: str = "any", title: str = "Add Step to Experiment"):
        super().__init__()
        self._initial         = initial_params or {}
        self._title           = title
        self._backend_ids     = backend_ids
        self._initial_gpu_pref = initial_gpu_pref

    def compose(self):
        defaults    = _load_defaults()
        gpu_options = [("any", "any")] + [(b, b) for b in self._backend_ids]
        yield Footer()
        with Vertical(id="spawn-dialog"):
            yield Label(self._title, id="spawn-title")
            yield Label("gpu")
            yield Select(gpu_options, value=self._initial_gpu_pref,
                         allow_blank=False, id="gpu-pref")
            with VerticalScroll(id="spawn-params"):
                for key, val in defaults.items():
                    current = self._initial.get(key, val)
                    yield Label(key)
                    yield Input(value=str(current), id=f"param-{key}")
            with Horizontal(id="spawn-buttons"):
                yield Button("Save", variant="success", id="step-confirm")
                yield Button("Cancel", variant="error", id="step-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "step-cancel":
            self.dismiss(None)
        elif event.button.id == "step-confirm":
            self.action_confirm()

    def action_confirm(self) -> None:
        params = _collect_params(self)
        params["gpu_preference"] = str(self.query_one("#gpu-pref", Select).value)
        self.dismiss(params)


class EditExperimentModal(ModalScreen):
    """Edit the first task of a queued experiment."""
    BINDINGS = [
        Binding("ctrl+s", "confirm", "Save"),
        Binding("escape", "dismiss", "Cancel"),
    ]

    def __init__(self, run_name: str, current_params: dict,
                 backend_ids: list[str], current_gpu_pref: str = "any"):
        super().__init__()
        self._run_name        = run_name
        self._current         = current_params
        self._backend_ids     = backend_ids
        self._current_gpu_pref = current_gpu_pref

    def compose(self):
        defaults = _load_defaults()
        gpu_options = [("any", "any")] + [(b, b) for b in self._backend_ids]
        yield Footer()
        with Vertical(id="spawn-dialog"):
            yield Label(f"Edit Experiment: {self._run_name}", id="spawn-title")
            yield Label("run_name")
            yield Input(value=self._run_name, id="run-name")
            yield Label("gpu")
            yield Select(gpu_options, value=self._current_gpu_pref, allow_blank=False, id="gpu-pref")
            with VerticalScroll(id="spawn-params"):
                for key, default_val in defaults.items():
                    val = self._current.get(key, default_val)
                    yield Label(key)
                    yield Input(value=str(val), id=f"param-{key}")
            with Horizontal(id="spawn-buttons"):
                yield Button("Save", variant="success", id="edit-confirm")
                yield Button("Cancel", variant="error", id="edit-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "edit-cancel":
            self.dismiss(None)
        elif event.button.id == "edit-confirm":
            self.action_confirm()

    def action_confirm(self) -> None:
        defaults = _load_defaults()
        params = {}
        for key, default_val in defaults.items():
            raw = self.query_one(f"#param-{key}", Input).value.strip()
            val = raw if raw else default_val
            params[key] = int(float(val)) if isinstance(default_val, int) else float(val)
        new_name = self.query_one("#run-name", Input).value.strip() or self._run_name
        gpu_pref = self.query_one("#gpu-pref", Select).value
        self.dismiss({"run_name": new_name, "params": params, "gpu_preference": str(gpu_pref)})


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

    def _find_completed_step(self, run_name: str, tasks: list[dict]
                              ) -> tuple[int, str | None]:
        """Scan checkpoint files to find the last completed step of a multi-step
        experiment. Returns (next_task_idx, checkpoint_path)."""
        last_completed = -1
        last_checkpoint = None
        for step_idx in range(len(tasks)):
            step_run_name = f"{run_name}_{step_idx + 1}"
            candidates = [c for c in self._checkpoints(step_run_name)
                          if "_latest" not in c.stem]
            if not candidates:
                break
            # Check if the .json companion has status=complete
            best = max(candidates, key=lambda p: p.stat().st_mtime)
            meta_path = best.with_suffix(".json")
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta.get("status") == "complete":
                    last_completed = step_idx
                    last_checkpoint = str(best)
                    continue
            # No status=complete — this step didn't finish
            break
        if last_completed >= 0:
            return last_completed + 1, last_checkpoint
        return 0, None

    def _exp_runs(self, base_run_name: str) -> list:
        """All TrainingRuns belonging to an experiment, matched by chain_experiment
        (multi-step) or run_name (single-step)."""
        return [r for r in self.runs
                if r.chain_experiment == base_run_name
                or (r.chain_experiment is None and r.run_name == base_run_name)]

    def _save_queue(self) -> None:
        with open(QUEUE_FILE, "w") as f:
            json.dump(self.experiment_queue, f, indent=2)

    def _load_queue(self) -> None:
        if not QUEUE_FILE.exists():
            return
        with open(QUEUE_FILE) as f:
            self.experiment_queue = json.load(f)

    def _spawn_chain_task(self, run) -> None:
        try:
            self._spawn_chain_task_inner(run)
        except Exception as exc:
            log_error("Chain step failed", exc=exc)
            self.notify(f"Chain step failed: {exc}", severity="error", timeout=10, markup=False)

    def _spawn_chain_task_inner(self, run) -> None:
        next_idx = run.chain_task_idx + 1
        exp = next((e for e in self.experiment_queue if e.get("run_name") == run.chain_experiment), None)
        if exp is None:
            self.notify(f"Chain: experiment '{run.chain_experiment}' not found in queue", severity="error", timeout=10)
            return
        tasks = exp.get("tasks") or [exp]
        if next_idx >= len(tasks):
            self.notify(f"Chain: no step {next_idx + 1} found for '{run.chain_experiment}'", severity="error", timeout=10)
            return
        candidates = [c for c in self._checkpoints(run.run_name) if "_latest" not in c.stem]
        checkpoint = str(max(candidates, key=lambda p: p.stat().st_mtime)) if candidates else None
        # Per-step GPU preference overrides experiment-level preference
        pref = tasks[next_idx].get("gpu_preference") or exp.get("gpu_preference", "any")
        used = {r.gpu_id for r in self.runs if r.status == "running" and r.gpu_id is not None}
        def _gpu_ready(g) -> bool:
            if g.index in used:
                return False
            if g.backend_id != "local":
                try:
                    s = g.status()
                    if s.instance_state != "running":
                        return False
                except Exception:
                    return False
            return True

        if pref == "any":
            gpu = next((g for g in self._gpus if _gpu_ready(g)), None)
        else:
            gpu = next((g for g in self._gpus if g.backend_id == pref and _gpu_ready(g)), None)
        if gpu is None:
            self.notify(f"No ready GPU available for chain step {next_idx + 1}", severity="error")
            return
        actual_run_name = f"{run.chain_experiment}_{next_idx + 1}"
        launch_config = {
            **tasks[next_idx],
            "run_name":   actual_run_name,
            "gpu_id":     gpu.index,
            "backend_id": gpu.backend_id,
            "checkpoint": checkpoint,
        }
        self.runs.remove(run)

        def _set_chain(new_run) -> None:
            new_run.chain_experiment  = run.chain_experiment
            new_run.chain_task_idx    = next_idx
            new_run.chain_total_tasks = run.chain_total_tasks
            self._save_state()
            self.notify(
                f"{actual_run_name}: step {next_idx + 1}/{run.chain_total_tasks} started — waiting for task to start",
                timeout=8,
            )

        self._on_spawn_result(launch_config, on_ready=_set_chain)

    def _refresh_queue_table(self) -> None:
        table = self.query_one("#queue-table", DataTable)
        table.clear()
        defaults = _load_defaults()
        self._queue_row_map: list[int] = []        # table row → experiment index
        self._queue_task_map: list[int] = []       # table row → task index within experiment

        for exp_idx, exp in enumerate(self.experiment_queue):
            tasks = exp.get("tasks") or [exp]
            run_name = exp.get("run_name", "-")

            exp_runs = self._exp_runs(run_name)
            done_run = next(
                (r for r in exp_runs if r.status in ("stopped", "killed", "error")), None,
            )
            running_run = next(
                (r for r in exp_runs if r.status == "running"), None,
            )
            completed_up_to = done_run.chain_task_idx if done_run is not None else -1
            running_idx = running_run.chain_task_idx if running_run is not None else None

            def step_mark(task_idx: int) -> str:
                if completed_up_to >= task_idx:
                    return " ✓"
                if running_idx == task_idx:
                    return " ▶"
                return ""

            def _gpu_label(task: dict, exp: dict) -> str:
                pref = task.get("gpu_preference") or exp.get("gpu_preference", "any")
                return "" if pref == "any" else pref

            task1 = tasks[0]
            changed = ", ".join(
                f"{k}={task1[k]}" for k in defaults
                if k in task1 and task1[k] != defaults[k]
            )
            table.add_row(run_name + step_mark(0), _gpu_label(task1, exp), changed or "(defaults)")
            self._queue_row_map.append(exp_idx)
            self._queue_task_map.append(0)

            # Show additional steps indented below the experiment header
            for step_i, task in enumerate(tasks[1:], start=2):
                changed_t = ", ".join(
                    f"{k}={task[k]}" for k in defaults
                    if k in task and task[k] != defaults[k]
                )
                is_last = step_i == len(tasks)
                prefix = "  └─" if is_last else "  ├─"
                task_idx = step_i - 1
                table.add_row(f"{prefix} step {step_i}{step_mark(task_idx)}",
                              _gpu_label(task, exp), changed_t or "(defaults)")
                self._queue_row_map.append(exp_idx)
                self._queue_task_map.append(task_idx)

        # Restore cursor to the row that corresponds to selected_queue_idx
        restore_row = next(
            (i for i, ei in enumerate(self._queue_row_map) if ei == self.selected_queue_idx),
            -1,
        )
        if restore_row >= 0:
            table.move_cursor(row=restore_row)

    # -----------------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------------
    def _backend_ids(self) -> list[str]:
        self._refresh_gpus()
        return list(dict.fromkeys(g.backend_id for g in self._gpus))

    def action_add_to_queue(self) -> None:
        self.push_screen(QueueModal(self._backend_ids()), self._on_queue_result)

    def _on_queue_result(self, config: dict | None) -> None:
        if config is None:
            return
        self.experiment_queue.append(config)
        self._save_queue()
        self._refresh_queue_table()
        self.notify(f"Queued {config['run_name']}", timeout=3)

    def _exp_idx_from_cursor(self) -> int | None:
        """Resolve the experiment index from the current queue table cursor."""
        table = self.query_one("#queue-table", DataTable)
        row_map = getattr(self, "_queue_row_map", [])
        cursor = table.cursor_row
        if not row_map or cursor < 0 or cursor >= len(row_map):
            return None
        return row_map[cursor]

    def action_add_step(self) -> None:
        idx = self._exp_idx_from_cursor()
        if idx is None:
            self.notify("No experiment selected.", severity="warning")
            return
        self.selected_queue_idx = idx
        exp         = self.experiment_queue[idx]
        default_gpu = exp.get("gpu_preference", "any")
        self.push_screen(
            AddStepModal(self._backend_ids(), initial_gpu_pref=default_gpu),
            self._on_add_step_result,
        )

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

    def action_edit_experiment(self) -> None:
        idx = self._exp_idx_from_cursor()
        if idx is None:
            self.notify("No experiment selected.", severity="warning")
            return
        self.selected_queue_idx = idx
        exp      = self.experiment_queue[idx]
        run_name = exp.get("run_name")
        tasks    = exp.get("tasks") or [exp]

        table    = self.query_one("#queue-table", DataTable)
        task_map = getattr(self, "_queue_task_map", [])
        cursor   = table.cursor_row
        task_idx = task_map[cursor] if task_map and 0 <= cursor < len(task_map) else 0

        # A step is submitted if any run for this experiment has reached or passed it
        submitted_up_to = -1
        existing = next(iter(self._exp_runs(run_name)), None)
        if existing is not None:
            submitted_up_to = existing.chain_task_idx

        if task_idx <= submitted_up_to:
            self.notify(
                f"Step {task_idx + 1} has already been submitted — cannot edit.",
                severity="warning",
            )
            return

        if task_idx == 0:
            # Edit experiment metadata + first step
            current_params   = {k: v for k, v in tasks[0].items() if k in _load_defaults()}
            current_gpu_pref = exp.get("gpu_preference", "any")
            self.push_screen(
                EditExperimentModal(run_name, current_params, self._backend_ids(), current_gpu_pref),
                self._on_edit_experiment_result,
            )
        else:
            # Edit an individual step
            current_params   = {k: v for k, v in tasks[task_idx].items() if k in _load_defaults()}
            current_gpu_pref = tasks[task_idx].get("gpu_preference", "any")
            self._editing_task_idx = task_idx
            self.push_screen(
                AddStepModal(self._backend_ids(), initial_params=current_params,
                             initial_gpu_pref=current_gpu_pref,
                             title=f"Edit Step {task_idx + 1}: {run_name}"),
                self._on_edit_step_result,
            )

    def _on_edit_step_result(self, params: dict | None) -> None:
        if params is None:
            return
        exp   = self.experiment_queue[self.selected_queue_idx]
        tasks = exp.get("tasks") or [exp]
        tasks[self._editing_task_idx] = params
        exp["tasks"] = tasks
        self._save_queue()
        self._refresh_queue_table()
        self.notify(f"Updated step {self._editing_task_idx + 1} of {exp['run_name']}", timeout=3)

    def _on_edit_experiment_result(self, result: dict | None) -> None:
        if result is None:
            return
        exp = self.experiment_queue[self.selected_queue_idx]
        tasks = exp.get("tasks") or [exp]
        tasks[0] = result["params"]
        exp["tasks"] = tasks
        exp["run_name"] = result["run_name"]
        exp["gpu_preference"] = result.get("gpu_preference", "any")
        self._save_queue()
        self._refresh_queue_table()
        self.notify(f"Updated {result['run_name']}", timeout=3)

    def action_remove_from_queue(self) -> None:
        idx = self._exp_idx_from_cursor()
        if idx is None:
            return
        self.selected_queue_idx = idx
        exp = self.experiment_queue[idx]
        run_name = exp.get("run_name")

        table = self.query_one("#queue-table", DataTable)
        task_map = getattr(self, "_queue_task_map", [])
        cursor = table.cursor_row
        task_idx = task_map[cursor] if task_map and 0 <= cursor < len(task_map) else 0

        if task_idx == 0:
            # Removing the whole experiment — block if running or already ran
            if self._exp_runs(run_name):
                self.notify(f"{run_name} has already started — cannot remove.", severity="warning")
                return
            self.experiment_queue.pop(self.selected_queue_idx)
            self.selected_queue_idx = min(self.selected_queue_idx, len(self.experiment_queue) - 1)
            self._save_queue()
            self._refresh_queue_table()
            self.notify(f"Removed experiment {run_name}", timeout=3)
        else:
            # Removing a specific step — block if that step has already run or is running
            existing = next(iter(self._exp_runs(run_name)), None)
            if existing is not None and existing.chain_task_idx >= task_idx:
                self.notify(f"Step {task_idx + 1} has already run — cannot remove.", severity="warning")
                return
            tasks = exp.get("tasks") or [exp]
            tasks.pop(task_idx)
            exp["tasks"] = tasks
            self._save_queue()
            self._refresh_queue_table()
            self.notify(f"Removed step {task_idx + 1} from {run_name}", timeout=3)

    def action_submit_queue(self) -> None:
        self._submit_selected()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "queue-submit":
            self._submit_selected()

    def _submit_selected(self) -> None:
        self._refresh_gpus()
        idx = self._exp_idx_from_cursor()
        if idx is None:
            self.notify("No experiment selected.", severity="warning")
            return
        table = self.query_one("#queue-table", DataTable)
        task_map = getattr(self, "_queue_task_map", [])
        cursor = table.cursor_row
        if task_map and 0 <= cursor < len(task_map) and task_map[cursor] > 0:
            self.notify("Select the experiment row (not a step) to submit the chain.", severity="warning")
            return
        self.selected_queue_idx = idx
        exp      = self.experiment_queue[idx]
        run_name = exp["run_name"]
        tasks    = exp.get("tasks") or [exp]
        multi    = len(tasks) > 1

        # Block if a run is already active for this experiment
        if any(r.status == "running" for r in self._exp_runs(run_name)):
            self.notify(f"{run_name} is already running.", severity="warning")
            return

        # Find the next step to run.
        # 1. Check tracked runs (may survive a restart via _restore_state)
        # 2. Fall back to scanning checkpoint files on disk
        start_task_idx = 0
        checkpoint     = None
        existing = next(
            (r for r in self._exp_runs(run_name) if r.status in ("stopped", "killed", "error")),
            None,
        )
        if existing is not None:
            # Only advance to the next step if the previous one completed cleanly.
            # If it errored or was killed, retry that same step.
            if existing.status == "stopped":
                start_task_idx = existing.chain_task_idx + 1
            else:
                start_task_idx = existing.chain_task_idx
            if start_task_idx >= len(tasks):
                self.notify(f"{run_name}: all {len(tasks)} steps already completed.", severity="warning")
                return
            candidates = [c for c in self._checkpoints(existing.run_name) if "_latest" not in c.stem]
            checkpoint = str(max(candidates, key=lambda p: p.stat().st_mtime)) if candidates else None
            self.runs.remove(existing)
        elif multi:
            # No tracked run — scan checkpoints to find last completed step
            start_task_idx, checkpoint = self._find_completed_step(run_name, tasks)
            if start_task_idx >= len(tasks):
                self.notify(f"{run_name}: all {len(tasks)} steps already completed.", severity="warning")
                return

        # Per-step GPU preference overrides experiment-level preference
        pref = tasks[start_task_idx].get("gpu_preference") or exp.get("gpu_preference", "any")
        used = {r.gpu_id for r in self.runs if r.status == "running" and r.gpu_id is not None}
        def _gpu_ready(g) -> bool:
            """True if the GPU slot is free and ready to accept a job."""
            if g.index in used:
                return False
            if g.backend_id != "local":
                try:
                    s = g.status()
                    if s.instance_state != "running":
                        return False
                except Exception:
                    return False
            return True

        if pref == "any":
            gpu = next((g for g in self._gpus if _gpu_ready(g)), None)
        else:
            gpu = next((g for g in self._gpus if g.backend_id == pref and _gpu_ready(g)), None)
        if gpu is None:
            log_error("No ready GPU available", preference=pref,
                      gpus=[f"{g.backend_id}:{g.index}" for g in self._gpus])
            self.notify(f"No ready GPU available{f' on {pref}' if pref != 'any' else ''}.", severity="error")
            return

        # Multi-step experiments get a _N suffix on the run name
        actual_run_name = f"{run_name}_{start_task_idx + 1}" if multi else run_name
        launch_config = {**tasks[start_task_idx], "run_name": actual_run_name,
                         "gpu_id": gpu.index, "backend_id": gpu.backend_id}
        if checkpoint:
            launch_config["checkpoint"] = checkpoint
        def _on_submitted(new_run) -> None:
            if multi:
                new_run.chain_experiment  = run_name
                new_run.chain_task_idx    = start_task_idx
                new_run.chain_total_tasks = len(tasks)
                self._save_state()
            step_label = f" (step {start_task_idx + 1}/{len(tasks)})" if multi else ""
            self.notify(f"Submitted {actual_run_name}{step_label} on GPU {gpu.index} — waiting for task to start", timeout=8)

        self._on_spawn_result(launch_config, on_ready=_on_submitted)
