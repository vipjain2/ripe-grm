"""Training dashboard — app entry point, tick loop, UI composition, and Tasks-tab actions."""

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from ripe_autotrain.compute_backend_client import GPU, JobConfig
from ripe_autotrain.dashboard_backend import init_backends, live_jobs
from ripe_autotrain.dashboard_experiments import (
    ExperimentsMixin, _load_defaults,
)
from ripe_autotrain.dashboard_train_runs import TrainingRun, SpawnModal
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button, DataTable, Footer, Header, Input, Label, Log,
    Static, TabbedContent, TabPane,
)

_TRAINING_DIR    = Path.cwd()
BACKEND_REGISTRY = _TRAINING_DIR / "compute_registry.json"

def _load_project_config() -> tuple[Path | None, Path | None]:
    cfg_file = _TRAINING_DIR / "dashboard_config.json"
    if not cfg_file.exists():
        return None, None
    cfg = json.loads(cfg_file.read_text())
    output_dir = (_TRAINING_DIR / cfg["output_dir"]) if "output_dir" in cfg else None
    log_dir    = (_TRAINING_DIR / cfg["log_dir"])    if "log_dir"    in cfg else None
    return output_dir, log_dir

_OUTPUT_DIR, _LOG_DIR = _load_project_config()

_backends, _backend_by_id, _gpus, N_GPUS = init_backends()

STATE_FILE  = _TRAINING_DIR / "runs_state.json"
CONFIG_FILE = _TRAINING_DIR / "dashboard_config.json"
LOG_DIR     = Path("/tmp/drl_runs")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
class Dashboard(App, ExperimentsMixin):
    TITLE = "AutoTrain"
    CSS = """
    TabbedContent { height: 1fr; }
    #main { height: 1fr; }
    #left-panel { width: 56; border: solid $primary; }
    #right-panel { width: 1fr; border: solid $primary; }
    #runs-table { height: 1fr; }
    #queue-table { height: 1fr; }
    #log-view { height: 1fr; }
    #gpu-panel { padding: 1 2; }
    #gpu-status { height: auto; }
    #queue-actions { height: 3; align: right middle; padding: 0 2; }
    #queue-submit { width: auto; min-width: 16; }
    SpawnModal { align: center middle; }
    #spawn-dialog {
        width: 52; height: 80vh;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    #tasks-container { height: 1fr; overflow-y: auto; }
    #spawn-title { text-align: center; text-style: bold; margin-bottom: 1; }
    #spawn-buttons { margin-top: 1; height: auto; align: center middle; }
    """

    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("n", "spawn",        "New run"),
        Binding("c", "continue_run", "Continue"),
        Binding("k", "kill",         "Kill selected"),
        Binding("d", "delete",       "Delete selected"),
        Binding("t", "tensorboard",  "TensorBoard"),
        Binding("r", "render",       "Render"),
        Binding("a", "add_to_queue",      "Add experiment"),
        Binding("x", "remove_from_queue", "Remove experiment"),
        Binding("e", "add_step",          "Extend"),
        Binding("s", "submit_queue",      "Submit queue"),
        Binding("ctrl+p", "command_palette", "Palette"),
        Binding("ctrl+c", "quit",    "Quit", priority=True),
        Binding("q", "quit",         "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.runs: list[TrainingRun] = []
        self.experiment_queue: list[dict] = []
        self.selected_idx: int = -1
        self.selected_run_name: str | None = None
        self._rebuilding_table: bool = False
        self.selected_queue_idx: int = -1
        self._active_tab: str = "tab-experiments"
        self._tb_process: subprocess.Popen | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent():
            with TabPane("Experiments", id="tab-experiments"):
                with Vertical():
                    yield DataTable(id="queue-table", cursor_type="row")
                    with Horizontal(id="queue-actions"):
                        yield Button("Submit", id="queue-submit", variant="success")
            with TabPane("Tasks", id="tab-tasks"):
                with Horizontal(id="main"):
                    with Vertical(id="left-panel"):
                        yield Label(f" MJX Training Runs  ({N_GPUS} GPUs)", markup=False)
                        yield DataTable(id="runs-table", cursor_type="row")
                    with Vertical(id="right-panel"):
                        yield Label(" Log Output", markup=False)
                        yield Log(id="log-view", auto_scroll=False)
            with TabPane("GPU Status", id="tab-gpu"):
                with Vertical(id="gpu-panel"):
                    yield Static("", id="gpu-status")
        yield Footer()

    def on_mount(self) -> None:
        runs_table = self.query_one("#runs-table", DataTable)
        runs_table.add_columns("Name", "Device", "ID", "Status", "Elapsed", "Steps", "Ckpt")

        queue_table = self.query_one("#queue-table", DataTable)
        queue_table.add_columns("Name", "Changed Params")

        self._load_config()
        self._load_state()
        self._load_existing_runs()
        self._load_queue()
        self.set_interval(1.0, self._tick)

    # -----------------------------------------------------------------------
    # State persistence
    # -----------------------------------------------------------------------
    def _load_config(self) -> None:
        if CONFIG_FILE.exists():
            config = json.loads(CONFIG_FILE.read_text())
            if "theme" in config:
                self.theme = config["theme"]

    def watch_theme(self, theme: str) -> None:
        CONFIG_FILE.write_text(json.dumps({"theme": theme}, indent=2))

    def _save_state(self) -> None:
        state = [
            {
                "run_name":          run.run_name,
                "gpu_id":            run.gpu_id,
                "log_file":          run.log_file,
                "start_time":        run.start_time,
                "steps":             run.steps,
                "steps_offset":      run.steps_offset,
                "chain_experiment":  run.chain_experiment,
                "chain_task_idx":    run.chain_task_idx,
                "chain_total_tasks": run.chain_total_tasks,
            }
            for run in self.runs if run.status == "running"
        ]
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    def _params_for_run(self, run_name: str) -> dict:
        """Read hyperparams from the run's latest JSON sidecar, fall back to _load_defaults()."""
        if _OUTPUT_DIR:
            candidates = list(_OUTPUT_DIR.glob(f"{run_name}*.json"))
        else:
            candidates = list(_TRAINING_DIR.rglob(f"{run_name}*.json"))
        if candidates:
            latest = max(candidates, key=lambda p: p.stat().st_mtime)
            with open(latest) as f:
                return json.load(f)
        return dict(_load_defaults())

    def _attach_running_processes(self) -> None:
        """Scan each GPU for training processes not yet tracked by the backend."""
        for gpu in _gpus:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", f"--id={gpu.index}",
                     "--query-compute-apps=pid",
                     "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                continue
            for pid_str in out.splitlines():
                pid_str = pid_str.strip()
                if not pid_str:
                    continue
                try:
                    pid = int(pid_str)
                    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\x00")
                    cmdline = [c.decode(errors="replace") for c in cmdline if c]
                    if "--run-name" not in cmdline:
                        continue
                    run_name = cmdline[cmdline.index("--run-name") + 1]
                    log_candidates = sorted(LOG_DIR.glob(f"{run_name}_*.log"),
                                           key=lambda p: p.stat().st_mtime)
                    log_file = str(log_candidates[-1]) if log_candidates else ""
                    backend = _backend_by_id.get(gpu.backend_id)
                    if backend:
                        backend.register_existing(
                            run_name=run_name, gpu_id=gpu.index,
                            log_file=log_file, job_id=str(pid),
                        )
                except Exception:
                    continue

    def _load_state(self) -> None:
        self._attach_running_processes()
        if not STATE_FILE.exists():
            return
        with open(STATE_FILE) as f:
            state = json.load(f)
        live = live_jobs(_backends)
        seen = {r.run_name for r in self.runs}
        for entry in state:
            run_name = entry["run_name"]
            handle   = live.get(run_name)
            run = TrainingRun(
                run_name          = run_name,
                gpu_id            = entry.get("gpu_id"),
                params            = self._params_for_run(run_name),
                handle            = handle,
                start_time        = entry.get("start_time", time.time()),
                status            = "running" if handle else "stopped",
                steps             = entry.get("steps", 0),
                steps_offset      = entry.get("steps_offset", 0),
                log_file          = entry.get("log_file", ""),
                chain_experiment  = entry.get("chain_experiment"),
                chain_task_idx    = entry.get("chain_task_idx", 0),
                chain_total_tasks = entry.get("chain_total_tasks", 1),
            )
            self.runs.append(run)
            seen.add(run_name)
            if handle and run.log_file and Path(run.log_file).exists():
                run.start_reader()

    def _load_existing_runs(self) -> None:
        seen = {r.run_name for r in self.runs}
        live = live_jobs(_backends)
        if _OUTPUT_DIR:
            ckpts = sorted(_OUTPUT_DIR.glob("*.msgpack"), key=lambda p: p.stat().st_mtime) if _OUTPUT_DIR.exists() else []
        else:
            ckpts = sorted(_TRAINING_DIR.rglob("*.msgpack"), key=lambda p: p.stat().st_mtime)
        for ckpt in ckpts:
            meta_path = ckpt.with_suffix(".json")
            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            stem = ckpt.stem
            run_name = meta.get("run_name") or re.sub(r"(_\d{8}_\d{4}|_final)$", "", stem)
            if run_name in seen:
                continue
            handle = live.get(run_name)
            run = TrainingRun(
                run_name=run_name,
                gpu_id=handle.gpu_id if handle else None,
                handle=handle,
                params=self._params_for_run(run_name),
                status="running" if handle else "stopped",
                log_file=handle.log_file if handle else "",
            )
            self.runs.append(run)
            if handle and run.log_file and Path(run.log_file).exists():
                run.start_reader()
            seen.add(run_name)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _free_gpu(self) -> "GPU | None":
        used = {r.gpu_id for r in self.runs if r.status == "running" and r.gpu_id is not None}
        return next((g for g in _gpus if g.index not in used), None)

    def _gpu_by_index(self, idx: int | None) -> "GPU | None":
        if idx is None:
            return None
        return next((g for g in _gpus if g.index == idx), None)

    def _checkpoints(self, run_name: str) -> list[Path]:
        if _OUTPUT_DIR or _LOG_DIR:
            candidates = list(_OUTPUT_DIR.glob(f"{run_name}*.msgpack")) if _OUTPUT_DIR else []
            if _LOG_DIR:
                candidates += list(_LOG_DIR.rglob(f"{run_name}*.msgpack"))
            return candidates
        return list(_TRAINING_DIR.rglob(f"{run_name}*.msgpack"))

    # -----------------------------------------------------------------------
    # Tick helpers
    # -----------------------------------------------------------------------
    def _process_run(self, run: TrainingRun) -> tuple[list[str], bool]:
        """Drain the run's log queue, parse state, detect errors/stops.

        Returns (new_lines, state_dirty).
        """
        state_dirty = False
        new_lines = run.drain_queue()
        run.log_lines.extend(new_lines)

        for line in new_lines:
            m = re.search(r"steps=\s*([\d,]+)", line)
            if m:
                run.steps = int(m.group(1).replace(",", ""))
                state_dirty = True

        for line in new_lines:
            if "Traceback (most recent call last)" in line or "Error:" in line:
                if run.is_alive():
                    run.terminate()
                    run.status = "error"
                    run.stopped_at = time.time()
                    run.log_lines.append("[dashboard] Error detected — process terminated.")
                    state_dirty = True

        if run.status == "running" and not run.is_alive():
            run.status = "stopped"
            run.stopped_at = time.time()
            state_dirty = True
            if run.chain_experiment and run.chain_task_idx + 1 < run.chain_total_tasks:
                self.call_after_refresh(lambda r=run: self._spawn_chain_task(r))

        return new_lines, state_dirty

    def _update_log_view(self, new_lines: list[str], log_widget: Log) -> None:
        """Write new lines to the log widget, preserving scroll position."""
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

    def _tick(self) -> None:
        log_widget  = self.query_one("#log-view", Log)
        state_dirty = False
        for run in self.runs:
            new_lines, dirty = self._process_run(run)
            state_dirty = state_dirty or dirty
            if run.run_name == self.selected_run_name:
                self._update_log_view(new_lines, log_widget)

        if state_dirty:
            self._save_state()

        self._refresh_table()
        self._refresh_gpu_status()

    def _refresh_gpu_status(self) -> None:
        lines = []
        for gpu in _gpus:
            try:
                s = gpu.status()
            except Exception:
                continue
            bar_filled = s.util_pct // 5
            bar        = "█" * bar_filled + "░" * (20 - bar_filled)
            temp_color = "red" if s.temp_c >= 80 else "yellow" if s.temp_c >= 70 else "green"
            cost_str   = f"  ${gpu.cost_per_hour:.2f}/hr" if gpu.cost_per_hour > 0 else ""
            lines.append(
                f"{gpu.name:<20s}  "
                f"[cyan]{bar}[/cyan] {s.util_pct:3d}%  "
                f"[{temp_color}]{s.temp_c:3d}°C[/{temp_color}]  "
                f"Mem: {s.mem_used_mb:5d}/{s.mem_total_mb:5d} MiB"
                f"{cost_str}"
            )
        if lines:
            self.query_one("#gpu-status", Static).update("\n".join(lines))

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

    def _has_checkpoint(self, run: TrainingRun) -> str:
        return "yes" if self._checkpoints(run.run_name) else "-"

    def _refresh_table(self) -> None:
        from textual.coordinate import Coordinate
        table = self.query_one("#runs-table", DataTable)
        sorted_runs = sorted(self.runs, key=lambda r: r.status != "running")
        current_keys = [r.run_name for r in sorted_runs]
        new_selected = -1

        if current_keys != getattr(self, "_table_run_keys", None):
            # Structure changed — full rebuild needed
            scroll_x, scroll_y = table.scroll_x, table.scroll_y
            self._rebuilding_table = True
            table.clear()
            for i, run in enumerate(sorted_runs):
                total = run.steps if run.steps else run.steps_offset
                table.add_row(
                    run.run_name, run.device_label,
                    (run.handle.id if run.handle else "-"),
                    run.status, run.elapsed,
                    f"{total:,}" if total else "-",
                    self._has_checkpoint(run),
                )
                if run.run_name == self.selected_run_name:
                    new_selected = i
            self._table_run_keys = current_keys
            if new_selected >= 0:
                table.move_cursor(row=new_selected)
            table.scroll_to(x=scroll_x, y=scroll_y, animate=False)
            self.call_after_refresh(lambda: setattr(self, "_rebuilding_table", False))
        else:
            # Same rows — update only changing cells in-place (no clear, no scroll disruption)
            for i, run in enumerate(sorted_runs):
                total = run.steps if run.steps else run.steps_offset
                table.update_cell_at(Coordinate(i, 3), run.status)
                table.update_cell_at(Coordinate(i, 4), run.elapsed)
                table.update_cell_at(Coordinate(i, 5), f"{total:,}" if total else "-")
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
    # Events
    # -----------------------------------------------------------------------
    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        tab_id = str(event.tab.id)
        if tab_id.endswith("tab-tasks"):
            self._active_tab = "tab-tasks"
        elif tab_id.endswith("tab-experiments"):
            self._active_tab = "tab-experiments"
            self._load_queue()
            self._refresh_queue_table()
        elif tab_id.endswith("tab-gpu"):
            self._active_tab = "tab-gpu"
        self.refresh_bindings()

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        tasks_only       = {"spawn", "continue_run", "kill", "delete", "tensorboard", "render"}
        experiments_only = {"add_to_queue", "remove_from_queue", "add_step", "submit_queue"}
        gpu_hidden       = tasks_only | experiments_only
        if self._active_tab == "tab-gpu" and action in gpu_hidden:
            return False
        if action in tasks_only and self._active_tab != "tab-tasks":
            return False
        if action in experiments_only and self._active_tab != "tab-experiments":
            return False
        return True

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if self._rebuilding_table:
            return
        if event.data_table.id == "runs-table" and event.cursor_row != self.selected_idx:
            self._show_run(event.cursor_row)
        elif event.data_table.id == "queue-table":
            self.selected_queue_idx = event.cursor_row

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "runs-table":
            self._show_run(event.cursor_row)
        elif event.data_table.id == "queue-table":
            self.selected_queue_idx = event.cursor_row

    # -----------------------------------------------------------------------
    # Actions — Tasks tab
    # -----------------------------------------------------------------------
    def action_spawn(self) -> None:
        gpu = self._free_gpu()
        self.push_screen(SpawnModal(gpu.index if gpu is not None else 0, N_GPUS), self._on_spawn_result)

    def _on_spawn_result(self, config: dict | None) -> None:
        if config is None:
            return
        gpu_id = config["gpu_id"]
        gpu    = next((g for g in _gpus if g.index == gpu_id), None)
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
        handle = gpu.submit(job_config)

        run = TrainingRun(
            run_name = config["run_name"],
            gpu_id   = gpu_id,
            params   = {k: v for k, v in config.items() if k in _load_defaults()},
            handle   = handle,
            log_file = str(log_file),
        )
        run.start_reader()
        self.runs.append(run)
        self._save_state()
        self._refresh_table()
        self.selected_idx = len(self.runs) - 1

    def _selected_run(self) -> "TrainingRun | None":
        if not self.selected_run_name:
            return None
        return next((r for r in self.runs if r.run_name == self.selected_run_name), None)

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
        config["gpu_id"] = run.gpu_id if run.gpu_id is not None else (fallback_gpu.index if fallback_gpu else 0)
        config["checkpoint"] = checkpoint
        self.runs.remove(run)
        self._on_spawn_result(config)
        self.runs[-1].steps_offset = self._last_step_from_tb(run.run_name)
        self.notify(f"Continuing {run.run_name} from {Path(checkpoint).name}", timeout=5)

    def action_kill(self) -> None:
        run = self._selected_run()
        if run is None:
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
        gpu = self._free_gpu() or (_gpus[0] if _gpus else None)
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

    # -----------------------------------------------------------------------
    # Quit
    # -----------------------------------------------------------------------
    async def action_command_palette(self) -> None:
        from textual.command import CommandPalette
        await self.push_screen(CommandPalette())

    def action_quit(self) -> None:
        self._save_state()
        if self._tb_process and self._tb_process.poll() is None:
            self._tb_process.terminate()
        self.exit()


def main():
    Dashboard().run()


if __name__ == "__main__":
    main()
