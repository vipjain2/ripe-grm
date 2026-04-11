"""Tests for kill/delete actions and spawn selection.

Regressions covered:
- After _on_spawn_result, selected_run_name must be set so kill/delete work.
- action_kill / action_delete notify the user when no run is selected.
- _submit_selected reads cursor row from the queue table, not stale
  selected_queue_idx (which gets corrupted by table.clear() cursor events).
"""

import json
import pytest

from conftest import FAKE_GPU, FakeHandle
from ripe_grm.dashboard_train_runs import TrainingRun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _running_run(name="exp_run") -> TrainingRun:
    handle = FakeHandle(run_name=name)
    run = TrainingRun(run_name=name, gpu_id=0, handle=handle, status="running")
    return run


# ---------------------------------------------------------------------------
# 1. selected_run_name is set after _on_spawn_result
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spawn_sets_selected_run_name(default_params_file):
    """Kill/delete require selected_run_name to be set.
    Regression: _on_spawn_result didn't set it, so the run was unselectable."""
    import ripe_grm.dashboard as dash_mod

    app = dash_mod.Dashboard()
    async with app.run_test(headless=True) as pilot:
        config = {
            "run_name": "my_run",
            "gpu_id": 0,
            **json.loads((default_params_file / "default_params.json").read_text()),
        }
        app._on_spawn_result(config)
        assert app.selected_run_name == "my_run", (
            "_on_spawn_result must set selected_run_name so kill/delete can find the run"
        )


# ---------------------------------------------------------------------------
# 2. action_kill works on a freshly spawned run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_kill_freshly_spawned_run(default_params_file):
    """After spawning a run, action_kill should terminate it."""
    import ripe_grm.dashboard as dash_mod

    app = dash_mod.Dashboard()
    async with app.run_test(headless=True) as pilot:
        run = _running_run("kill_target")
        app.runs.append(run)
        app.selected_run_name = "kill_target"

        app.action_kill()
        await pilot.pause()

        assert run.status == "killed", "action_kill must set run.status to 'killed'"
        assert not run.handle.is_running(), "action_kill must cancel the handle"


# ---------------------------------------------------------------------------
# 3. action_delete removes the run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_removes_run(default_params_file):
    """After selecting a run, action_delete should remove it from self.runs."""
    import ripe_grm.dashboard as dash_mod

    app = dash_mod.Dashboard()
    async with app.run_test(headless=True) as pilot:
        run = _running_run("delete_target")
        app.runs.append(run)
        app.selected_run_name = "delete_target"

        app.action_delete()
        await pilot.pause()

        assert not any(r.run_name == "delete_target" for r in app.runs), (
            "action_delete must remove the run from app.runs"
        )


# ---------------------------------------------------------------------------
# 4. kill with no run selected notifies instead of silently doing nothing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_kill_no_selection_notifies(default_params_file):
    """action_kill with nothing selected should notify the user."""
    import ripe_grm.dashboard as dash_mod

    app = dash_mod.Dashboard()
    async with app.run_test(headless=True) as pilot:
        notifications = []
        app.notify = lambda msg, **kw: notifications.append(msg)
        app.selected_run_name = None

        app.action_kill()
        await pilot.pause()

        assert any("No run selected" in n for n in notifications), (
            "action_kill with no selection must notify the user — got: " + str(notifications)
        )


# ---------------------------------------------------------------------------
# 5. _submit_selected uses table cursor, not stale selected_queue_idx
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_uses_table_cursor_not_stale_index(default_params_file):
    """Regression: table.clear() reset selected_queue_idx to 0 via cursor events,
    so submitting always submitted the first experiment regardless of visual selection."""
    import ripe_grm.dashboard as dash_mod

    task_params = json.loads((default_params_file / "default_params.json").read_text())

    app = dash_mod.Dashboard()
    async with app.run_test(headless=True) as pilot:
        app.experiment_queue = [
            {"run_name": "exp_a", "tasks": [task_params]},
            {"run_name": "exp_b", "tasks": [task_params]},
            {"run_name": "exp_c", "tasks": [task_params]},
        ]
        app._refresh_queue_table()
        await pilot.pause()

        # Move the table cursor to exp_c (row 2)
        from textual.widgets import DataTable
        table = app.query_one("#queue-table", DataTable)
        table.move_cursor(row=2)
        await pilot.pause()

        # Simulate the stale index the bug left behind (table.clear() reset it to 0)
        app.selected_queue_idx = 0

        spawned = []
        app._on_spawn_result = lambda cfg: spawned.append(cfg["run_name"]) if cfg else None

        app._submit_selected()

        assert spawned == ["exp_c"], (
            f"Expected exp_c to be submitted (cursor on row 2), got {spawned}. "
            "_submit_selected must read cursor from table, not stale selected_queue_idx."
        )
