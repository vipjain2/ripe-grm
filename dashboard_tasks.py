"""Tasks tab — TasksTab widget, TasksMixin, and all Tasks-tab actions."""

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from ripe_autotrain.compute_backend_client import JobConfig
from ripe_autotrain.dashboard_experiments import _load_defaults
from ripe_autotrain.dashboard_log import log_error
from ripe_autotrain.dashboard_train_runs import TrainingRun, SpawnModal
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import DataTable, Label, Log

_TRAINING_DIR = Path.cwd()
LOG_DIR       = Path("/tmp/drl_runs")

def _load_project_config() -> tuple[Path | None, Path | None]:
    cfg_file = _TRAINING_DIR / "dashboard_config.json"
    if not cfg_file.exists():
        return None, None
    cfg = json.loads(cfg_file.read_text())
    output_dir = (_TRAINING_DIR / cfg["output_dir"]) if "output_dir" in cfg else None
    log_dir    = (_TRAINING_DIR / cfg["log_dir"])    if "log_dir"    in cfg else None
    return output_dir, log_dir

_OUTPUT_DIR, _LOG_DIR = _load_project_config()


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
class TasksTab(Widget):
    DEFAULT_CSS = "TasksTab { height: 1fr; }"
    BINDINGS = [
        Binding("q", "app.quit",         "Quit"),
        Binding("n", "app.spawn",        "New run"),
        Binding("c", "app.continue_run", "Continue"),
        Binding("k", "app.kill",         "Kill"),
        Binding("d", "app.delete",       "Delete"),
        Binding("t", "app.tensorboard",  "TensorBoard"),
        Binding("r", "app.render",       "Render"),
    ]

    def __init__(self, n_gpus: int):
        super().__init__()
        self._n_gpus = n_gpus

    def compose(self) -> ComposeResult:
        with Horizontal(id="main"):
            with Vertical(id="left-panel"):
                yield Label(f" MJX Training Runs  ({self._n_gpus} GPUs)", markup=False)
                yield DataTable(id="runs-table", cursor_type="row")
            with Vertical(id="right-panel"):
                yield Label(" Log Output", markup=False)
                yield Log(id="log-view", auto_scroll=False)


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------
class TasksMixin:
    """Mixin for Dashboard providing all Tasks-tab logic.

    Relies on the host class (Dashboard) providing:
        self.runs, self.selected_run_name, self.selected_idx,
        self._rebuilding_table, self._gpus,
        self._free_gpu(), self._gpu_by_index(), self._save_state(),
        self.notify(), self.push_screen(), self.query_one()
    """

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _checkpoints(self, run_name: str) -> list[Path]:
        if _OUTPUT_DIR or _LOG_DIR:
            candidates = list(_OUTPUT_DIR.glob(f"{run_name}*.msgpack")) if _OUTPUT_DIR else []
            if _LOG_DIR:
                candidates += list(_LOG_DIR.rglob(f"{run_name}*.msgpack"))
            return candidates
        return list(_TRAINING_DIR.rglob(f"{run_name}*.msgpack"))

    def _params_for_run(self, run_name: str) -> dict:
        if _OUTPUT_DIR:
            candidates = list(_OUTPUT_DIR.glob(f"{run_name}*.json"))
        else:
            candidates = list(_TRAINING_DIR.rglob(f"{run_name}*.json"))
        if candidates:
            latest = max(candidates, key=lambda p: p.stat().st_mtime)
            with open(latest) as f:
                return json.load(f)
        return dict(_load_defaults())

    def _selected_run(self) -> "TrainingRun | None":
        if not self.selected_run_name:
            return None
        return next((r for r in self.runs if r.run_name == self.selected_run_name), None)

    def _last_step_from_tb(self, run_name: str) -> int:
        try:
            from tbparse import SummaryReader
            if _LOG_DIR:
                log_dir = _LOG_DIR / run_name
            else:
                parents = [p.parent for p in _TRAINING_DIR.rglob("events.out.tfevents.*")
                           if run_name in str(p)]
                log_dir = parents[0] if parents else None
            if not log_dir or not Path(log_dir).exists():
                return 0
            df = SummaryReader(str(log_dir)).scalars
            return int(df["step"].max()) if not df.empty else 0
        except Exception:
            return 0

    # -----------------------------------------------------------------------
    # Tick helpers
    # -----------------------------------------------------------------------
    def _process_run(self, run: TrainingRun) -> tuple[list[str], bool]:
        """Drain the run's log queue, parse state, detect errors/stops."""
        state_dirty = False
        new_lines = run.drain_queue()
        run.log_lines.extend(new_lines)

        for line in new_lines:
            m = re.search(r"steps=\s*([\d,]+)", line)
            if m:
                run.steps = int(m.group(1).replace(",", ""))
                state_dirty = True

        if run.status == "running":
            for line in new_lines:
                if "Traceback (most recent call last)" in line:
                    if run.is_alive():
                        # Process still running — Traceback may be from historical log
                        # content (e.g. a previous failed attempt). Don't act yet.
                        break
                    log_error("Traceback detected in log; process not alive",
                              run_name=run.run_name, gpu_id=run.gpu_id)
                    run.status = "error"
                    run.stopped_at = time.time()
                    run.log_lines.append("[dashboard] Error detected.")
                    state_dirty = True
                    break

        if run.status == "running":
            run.alive_failures = 0 if run.is_alive() else run.alive_failures + 1

        if run.status == "running" and run.alive_failures >= 3:
            run.status = "stopped"
            run.stopped_at = time.time()
            state_dirty = True
            if run.chain_experiment and run.chain_task_idx + 1 < run.chain_total_tasks:
                self.notify(
                    f"{run.run_name}: step {run.chain_task_idx + 1} done, launching step {run.chain_task_idx + 2}…",
                    timeout=5,
                )
                self.call_after_refresh(lambda r=run: self._spawn_chain_task(r))

        return new_lines, state_dirty

    def _update_log_view(self, new_lines: list[str], log_widget: Log) -> None:
        if not new_lines:
            return
        at_bottom = log_widget.scroll_y >= log_widget.virtual_size.height - log_widget.size.height - 3
        scroll_y = log_widget.scroll_y
        for line in new_lines:
            log_widget.write_line(line)
        if at_bottom:
            log_widget.scroll_end(animate=False)
        else:
            log_widget.scroll_to(y=scroll_y, animate=False)

    def _refresh_table(self) -> None:
        from textual.coordinate import Coordinate
        table = self.query_one("#runs-table", DataTable)
        sorted_runs = sorted(self.runs, key=lambda r: r.status != "running")
        current_keys = [r.run_name for r in sorted_runs]
        new_selected = -1

        if current_keys != getattr(self, "_table_run_keys", None):
            scroll_x, scroll_y = table.scroll_x, table.scroll_y
            self._rebuilding_table = True
            table.clear()
            for i, run in enumerate(sorted_runs):
                table.add_row(
                    run.run_name, run.device_label,
                    (run.handle.id if run.handle else "-"),
                    run.status, run.elapsed,
                )
                if run.run_name == self.selected_run_name:
                    new_selected = i
            self._table_run_keys = current_keys
            if new_selected >= 0:
                table.move_cursor(row=new_selected)
            table.scroll_to(x=scroll_x, y=scroll_y, animate=False)
            self.call_after_refresh(lambda: setattr(self, "_rebuilding_table", False))
        else:
            for i, run in enumerate(sorted_runs):
                table.update_cell_at(Coordinate(i, 3), run.status)
                table.update_cell_at(Coordinate(i, 4), run.elapsed)
                if run.run_name == self.selected_run_name:
                    new_selected = i
            self.selected_idx = new_selected

    def _show_run(self, idx: int) -> None:
        sorted_runs = sorted(self.runs, key=lambda r: r.status != "running")
        if idx >= len(sorted_runs):
            return
        run = sorted_runs[idx]
        if run.run_name == self.selected_run_name:
            return
        self.selected_idx = idx
        self.selected_run_name = run.run_name
        log = self.query_one("#log-view", Log)
        log.clear()
        for line in run.log_lines[-500:]:
            log.write_line(line)
        log.scroll_end(animate=False)

    # -----------------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------------
    def action_spawn(self) -> None:
        gpu = self._free_gpu()
        n_gpus = len(self._gpus)
        self.push_screen(SpawnModal(gpu.index if gpu is not None else 0, n_gpus), self._on_spawn_result)

    def _on_spawn_result(self, config: dict | None) -> None:
        if config is None:
            return
        gpu_id     = config["gpu_id"]
        backend_id = config.get("backend_id")
        if backend_id:
            gpu = next((g for g in self._gpus
                        if g.index == gpu_id and g.backend_id == backend_id), None)
        else:
            gpu = next((g for g in self._gpus if g.index == gpu_id), None)
        if gpu is None:
            self.notify(f"GPU {gpu_id} not found.", severity="error")
            return

        ts       = time.strftime("%Y%m%d_%H%M%S")
        log_file = LOG_DIR / f"{config['run_name']}_{ts}.log"

        job_config = JobConfig(
            run_name   = config["run_name"],
            script     = "train.py",
            params     = {k: v for k, v in config.items() if k in _load_defaults()},
            checkpoint = config.get("checkpoint"),
            log_file   = str(log_file),
        )
        self.notify(f"Syncing files to {gpu.name}…", timeout=4)
        try:
            handle = gpu.submit(job_config)
        except Exception as e:
            log_error("gpu.submit failed", exc=e, run_name=config["run_name"],
                      gpu_id=gpu_id, backend_id=getattr(gpu, "backend_id", "?"))
            self.notify(f"Submit failed: {e}", severity="error", markup=False)
            return

        run = TrainingRun(
            run_name = config["run_name"],
            gpu_id   = gpu_id,
            params   = {k: v for k, v in config.items() if k in _load_defaults()},
            handle   = handle,
            log_file = str(log_file),
        )
        run.start_reader()
        self.runs.append(run)
        self.selected_run_name = run.run_name
        self._save_state()
        self._refresh_table()
        self.selected_idx = len(self.runs) - 1

    def action_continue_run(self) -> None:
        run = self._selected_run()
        if run is None:
            self.notify("No run selected.", severity="error")
            return
        if run.status == "running":
            self.notify("Run is still active — kill it first.", severity="error")
            return
        candidates = self._checkpoints(run.run_name)
        if not candidates:
            self.notify(f"No checkpoint found for {run.run_name}.", severity="error")
            return
        checkpoint = str(max(candidates, key=lambda p: p.stat().st_mtime))
        config = self._params_for_run(run.run_name)
        config["run_name"]   = run.run_name
        fallback_gpu = self._free_gpu()
        config["gpu_id"]     = run.gpu_id if run.gpu_id is not None else (fallback_gpu.index if fallback_gpu else 0)
        config["checkpoint"] = checkpoint
        self.runs.remove(run)
        self._on_spawn_result(config)
        self.runs[-1].steps_offset = self._last_step_from_tb(run.run_name)
        self.notify(f"Continuing {run.run_name} from {Path(checkpoint).name}", timeout=5)

    def action_kill(self) -> None:
        run = self._selected_run()
        if run is None:
            self.notify("No run selected.", severity="warning")
            return
        if run.status == "running":
            run.terminate()
            run.status = "killed"
            run.stopped_at = time.time()
            run.log_lines.append("[dashboard] Process terminated.")
            self._save_state()
            self._refresh_table()
            self.query_one("#log-view", Log).write_line("[dashboard] Process terminated.")

    def action_delete(self) -> None:
        run = self._selected_run()
        if run is None:
            self.notify("No run selected.", severity="warning")
            return
        if run.status == "running":
            run.terminate()
        self.runs.remove(run)
        self.selected_run_name = None
        self._save_state()
        self._refresh_table()
        self.query_one("#log-view", Log).clear()

    def action_render(self) -> None:
        run = self._selected_run()
        if run is None:
            self.notify("No run selected.", severity="error")
            return
        candidates = self._checkpoints(run.run_name)
        if not candidates:
            self.notify(f"No .msgpack found for {run.run_name}.", severity="error")
            return
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        gpu  = self._free_gpu() or (self._gpus[0] if self._gpus else None)
        env  = os.environ.copy()
        if gpu:
            env.update(gpu.env_vars())
        proc = subprocess.Popen(
            [sys.executable, "render.py", "--model", str(latest)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=_TRAINING_DIR, env=env,
        )

        def _read_render(q):
            for line in proc.stdout:
                q.put(line.rstrip())
            q.put(None)

        threading.Thread(target=_read_render, args=(run.log_queue,), daemon=True).start()
        self.notify(f"Rendering {run.run_name} — {latest.name}", timeout=5)

    def action_tensorboard(self) -> None:
        if self._tb_process and self._tb_process.poll() is None:
            self._tb_process.terminate()
            self._tb_process = None
            self.notify("TensorBoard stopped.", severity="warning")
        else:
            self._tb_process = subprocess.Popen(
                [sys.executable, "-m", "tensorboard.main", "--logdir", str(_LOG_DIR) if _LOG_DIR else "."],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                cwd=_TRAINING_DIR,
            )
            self.notify("TensorBoard started at http://localhost:6006", timeout=5)
