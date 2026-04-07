"""Training dashboard — app entry point, tick loop, UI composition, and state management."""

import json
import re
import subprocess
import time
from pathlib import Path

from ripe_autotrain.compute_backend_client import GPU, JobConfig
from ripe_autotrain.dashboard_backend import init_backends, live_jobs
from ripe_autotrain.dashboard_experiments import ExperimentsMixin, _load_defaults
from ripe_autotrain.dashboard_tasks import TasksMixin, TasksTab, LOG_DIR
from ripe_autotrain.dashboard_train_runs import TrainingRun
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import (
    Button, DataTable, Footer, Header,
    Static, TabbedContent, TabPane,
)

_TRAINING_DIR = Path.cwd()

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


# ---------------------------------------------------------------------------
# Experiments tab widget
# ---------------------------------------------------------------------------
class ExperimentsTab(Widget):
    DEFAULT_CSS = "ExperimentsTab { height: 1fr; }"
    BINDINGS = [
        Binding("q", "app.quit",                "Quit"),
        Binding("a", "app.add_to_queue",        "Add"),
        Binding("r", "app.remove_from_queue",   "Remove"),
        Binding("e", "app.edit_experiment",     "Edit"),
        Binding("x", "app.add_step",            "Extend"),
    ]

    def compose(self) -> ComposeResult:
        from textual.containers import Horizontal
        with Vertical():
            yield DataTable(id="queue-table", cursor_type="row")
            with Horizontal(id="queue-actions"):
                yield Button("Submit", id="queue-submit", variant="success")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
class GPUStatusTab(Widget):
    DEFAULT_CSS = "GPUStatusTab { height: 1fr; }"
    BINDINGS = [Binding("q", "app.quit", "Quit")]
    can_focus = True

    def compose(self) -> ComposeResult:
        with Vertical(id="gpu-panel"):
            yield Static("", id="gpu-status")


class Dashboard(App, TasksMixin, ExperimentsMixin):
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
    #spawn-params { height: 1fr; }
    #spawn-title { text-align: center; text-style: bold; margin-bottom: 1; }
    #spawn-buttons { margin-top: 1; height: auto; align: center middle; }
    """

    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("ctrl+p", "command_palette", "Palette"),
        Binding("ctrl+c", "quit", "Quit", priority=True, show=False),
    ]

    def __init__(self):
        super().__init__()
        self.runs: list[TrainingRun] = []
        self.experiment_queue: list[dict] = []
        self.selected_idx: int = -1
        self.selected_run_name: str | None = None
        self._rebuilding_table: bool = False
        self.selected_queue_idx: int = -1
        self._tb_process: subprocess.Popen | None = None
        self._gpus = _gpus

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent():
            with TabPane("Experiments", id="tab-experiments"):
                yield ExperimentsTab()
            with TabPane("Tasks", id="tab-tasks"):
                yield TasksTab(N_GPUS)
            with TabPane("GPU Status", id="tab-gpu"):
                yield GPUStatusTab()
        yield Footer()

    def on_mount(self) -> None:
        runs_table = self.query_one("#runs-table", DataTable)
        runs_table.add_columns("Name", "Device", "ID", "Status", "Elapsed")

        queue_table = self.query_one("#queue-table", DataTable)
        queue_table.add_columns("Name", "GPU", "Changed Params")

        self._load_config()
        self._load_state()
        self._load_existing_runs()
        self._load_queue()
        self.set_interval(1.0, self._tick)

    # -----------------------------------------------------------------------
    # Config / state persistence
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
            run_name = meta.get("run_name") or re.sub(r"(_\d{8}_\d{4}|_final|_latest)$", "", stem)
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
        self._refresh_gpus()
        used = {r.gpu_id for r in self.runs if r.status == "running" and r.gpu_id is not None}
        return next((g for g in self._gpus if g.index not in used), None)

    def _gpu_by_index(self, idx: int | None) -> "GPU | None":
        if idx is None:
            return None
        found = next((g for g in self._gpus if g.index == idx), None)
        if found is None:
            self._refresh_gpus()
            found = next((g for g in self._gpus if g.index == idx), None)
        return found

    # -----------------------------------------------------------------------
    # Tick
    # -----------------------------------------------------------------------
    def _tick(self) -> None:
        from textual.widgets import Log
        log_widget  = self.query_one("#log-view", Log)
        state_dirty = False
        for run in self.runs:
            new_lines, dirty = self._process_run(run)
            state_dirty = state_dirty or dirty
            if run.run_name == self.selected_run_name:
                self._update_log_view(new_lines, log_widget)

        if state_dirty:
            self._save_state()

        now = time.time()
        if now - getattr(self, "_last_cloud_poll", 0) >= 60:
            self._last_cloud_poll = now
            import threading
            threading.Thread(target=self._poll_cloud_runs, daemon=True).start()

        self._refresh_table()

    def _poll_cloud_runs(self) -> None:
        """Detect cloud jobs tracked by the backend but not yet in self.runs."""
        seen = {r.run_name for r in self.runs}
        for handle in (j for b in _backends for j in b.running_jobs()):
            if handle.run_name in seen or not handle.run_name:
                continue
            gpu = self._gpu_by_index(handle.gpu_id)
            run = TrainingRun(
                run_name = handle.run_name,
                gpu_id   = handle.gpu_id,
                params   = self._params_for_run(handle.run_name),
                handle   = handle,
                log_file = handle.log_file,
                status   = "running",
            )
            if handle.log_file:
                run.start_reader()
            self.runs.append(run)
            seen.add(handle.run_name)
            self.notify(f"Detected cloud run: {handle.run_name}", timeout=5)

    def _refresh_gpus(self) -> None:
        """Discover available GPUs from all backends (on-demand, call before scheduling)."""
        try:
            self._gpus = [g for b in _backends for g in b.discover_gpus()]
        except Exception:
            pass

    def _refresh_gpu_status(self) -> None:
        """Fetch GPU status in a background thread and update the GPU Status tab."""
        import threading
        threading.Thread(target=self._refresh_gpu_status_bg, daemon=True).start()

    def _refresh_gpu_status_bg(self) -> None:
        self._refresh_gpus()
        lines = []
        for gpu in self._gpus:
            try:
                s = gpu.status()
            except Exception as e:
                lines.append(f"{gpu.name:<20s}  [red]{e}[/red]")
                continue
            if s.mem_total_mb == 0:
                cost_str = f"  ${gpu.cost_per_hour:.2f}/hr" if gpu.cost_per_hour > 0 else ""
                lines.append(f"{gpu.name:<20s}  [dim]idle — no instance{cost_str}[/dim]")
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
            self.call_from_thread(
                self.query_one("#gpu-status", Static).update, "\n".join(lines)
            )

    # -----------------------------------------------------------------------
    # Events
    # -----------------------------------------------------------------------
    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        tab_id = str(event.tab.id)
        if tab_id.endswith("tab-experiments"):
            self._load_queue()
            self._refresh_queue_table()
            self.query_one("#queue-table", DataTable).focus()
        elif tab_id.endswith("tab-tasks"):
            self.query_one("#runs-table", DataTable).focus()
        elif tab_id.endswith("tab-gpu"):
            self.query_one(GPUStatusTab).focus()
            self._refresh_gpu_status()

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
