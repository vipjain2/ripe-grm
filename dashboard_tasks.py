"""Tasks tab — TasksTab widget, TasksMixin, and all Tasks-tab actions."""

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from ripe_grm.compute_backend_client import JobConfig
from ripe_grm.dashboard_experiments import _load_defaults
from ripe_grm.dashboard_log import log_error
from ripe_grm.dashboard_train_runs import TrainingRun
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
        self._free_gpu(), self._save_state(),
        self.notify(), self.push_screen(), self.query_one()
    """

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _checkpoints(self, run_name: str) -> list[Path]:
        """Find checkpoint files for a specific run name.

        Uses {run_name}_*.msgpack to avoid prefix collisions
        (e.g. 'ppo_exp8_2' matching 'ppo_exp8_20260410...').
        """
        def _gather(root: Path, recurse: bool = False) -> list[Path]:
            g = root.rglob if recurse else root.glob
            return list(g(f"{run_name}_*.msgpack")) + list(g(f"{run_name}.msgpack"))
        if _OUTPUT_DIR or _LOG_DIR:
            candidates = _gather(_OUTPUT_DIR) if _OUTPUT_DIR else []
            if _LOG_DIR:
                candidates += _gather(_LOG_DIR, recurse=True)
            return candidates
        return _gather(_TRAINING_DIR, recurse=True)

    def _stopped_summary(self, run: "TrainingRun") -> list[str]:
        """Build summary lines for a stopped/killed/error run."""
        lines = ["", "─── Run Summary ───"]
        # Latest checkpoint
        candidates = [c for c in self._checkpoints(run.run_name) if "_latest" not in c.stem]
        if candidates:
            latest_ckpt = max(candidates, key=lambda p: p.stat().st_mtime)
            lines.append(f"  checkpoint: {latest_ckpt.name}")
        else:
            lines.append("  checkpoint: (none)")
        # Trained iterations
        trained = run.steps
        if trained == 0:
            lines.append("  trained iterations: 0 (no log data)")
        else:
            lines.append(f"  trained iterations: {trained:,}")
        # Params — only show tunable keys (those present in defaults), so we
        # don't spuriously mark internal fields like obs_dim as overrides.
        params_file = self._params_file_for_run(run.run_name)
        if params_file is None:
            params = dict(_load_defaults())
            lines.append("  params: (not downloaded — showing defaults)")
        else:
            with open(params_file) as f:
                params = json.load(f)
        defaults = dict(_load_defaults())
        for k, default in defaults.items():
            if k not in params:
                continue
            v = params[k]
            marker = "" if v == default else "  <--"
            lines.append(f"  {k}: {v}{marker}")
        lines.append("───────────────────")
        return lines

    def _params_file_for_run(self, run_name: str) -> Path | None:
        """Return the latest params JSON for a run, or None if not found."""
        if _OUTPUT_DIR:
            candidates = list(_OUTPUT_DIR.glob(f"{run_name}_*.json")) + list(_OUTPUT_DIR.glob(f"{run_name}.json"))
        else:
            candidates = list(_TRAINING_DIR.rglob(f"{run_name}_*.json")) + list(_TRAINING_DIR.rglob(f"{run_name}.json"))
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _params_for_run(self, run_name: str) -> dict:
        params_file = self._params_file_for_run(run_name)
        if params_file is None:
            return dict(_load_defaults())
        with open(params_file) as f:
            return json.load(f)

    def _selected_run(self) -> "TrainingRun | None":
        if not self.selected_run_name:
            return None
        return next((r for r in self.runs if r.run_name == self.selected_run_name), None)

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
                    if run.chain_experiment:
                        self._update_disk_completed(run.chain_experiment)
                    state_dirty = True
                    break

        if run.status == "running":
            alive = run.is_alive()
            if not alive and run.alive_failures == 0:
                # First failure — check if the backend is tracking a different
                # run name (e.g. a prior process the dashboard didn't know about).
                actual = run.actual_run_name()
                if actual and actual != run.run_name:
                    log_error("Backend tracking different run than dashboard",
                              dashboard=run.run_name, backend=actual,
                              gpu_id=run.gpu_id)
                    run.run_name = actual
                    run.log_lines.append(
                        f"[dashboard] GPU slot is running '{actual}', updating run name.")
                    state_dirty = True
                    alive = True  # re-check on next tick with corrected name
            run.alive_failures = 0 if alive else run.alive_failures + 1

        if run.status == "running" and run.alive_failures >= 3:
            run.status = "stopped"
            run.stopped_at = time.time()
            state_dirty = True
            if run.chain_experiment:
                self._update_disk_completed(run.chain_experiment)
            if run.chain_experiment and run.chain_task_idx + 1 < run.chain_total_tasks:
                self.notify(
                    f"{run.run_name}: step {run.chain_task_idx + 1} done, launching step {run.chain_task_idx + 2}…",
                    timeout=5,
                )
                self.call_after_refresh(lambda r=run: self._spawn_chain_task(r))
            else:
                self.notify(f"{run.run_name}: training complete", timeout=8)

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
        # Add summary for restored stopped runs that don't have one yet
        if run.status in ("stopped", "killed", "error") and not any(
            "Run Summary" in l for l in run.log_lines[-25:]
        ):
            run.log_lines.extend(self._stopped_summary(run))
        log = self.query_one("#log-view", Log)
        log.clear()
        for line in run.log_lines[-500:]:
            log.write_line(line)
        log.scroll_end(animate=False)

    # -----------------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------------
    def _on_spawn_result(self, config: dict | None,
                         on_ready: "callable[[TrainingRun], None] | None" = None,
                         ) -> None:
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

        def _do_submit() -> None:
            try:
                handle = gpu.submit(job_config)
            except Exception as e:
                log_error("gpu.submit failed", exc=e, run_name=config["run_name"],
                          gpu_id=gpu_id, backend_id=getattr(gpu, "backend_id", "?"))
                self.call_from_thread(
                    self.notify, f"Submit failed: {e}",
                    severity="error", markup=False)
                return

            def _finish() -> None:
                # Dedup: poll thread may have already discovered this run
                existing = next(
                    (r for r in self.runs if r.run_name == config["run_name"]),
                    None,
                )
                if existing is not None:
                    existing.handle   = handle
                    existing.log_file = str(log_file)
                    existing.gpu_id   = gpu_id
                    run = existing
                else:
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
                if on_ready:
                    on_ready(run)

            self.call_from_thread(_finish)

        self.run_worker(_do_submit, thread=True)

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
