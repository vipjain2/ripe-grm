# Dashboard Tests

Run with:
```
pytest test/
```

---

## test_dashboard_kill_delete.py

### test_spawn_sets_selected_run_name
After `_on_spawn_result` creates a new training run, `app.selected_run_name` must be
set to the new run's name. Without this, `_selected_run()` returns `None` and both
`action_kill` and `action_delete` silently do nothing on the freshly spawned run.

### test_kill_freshly_spawned_run
`action_kill` on a selected running run must set `run.status = "killed"` and call
`handle.cancel()`. Verifies the full kill path works end-to-end once `selected_run_name`
is correctly set.

### test_delete_removes_run
`action_delete` on a selected run must remove it from `app.runs`. Also verifies that
a running run is terminated before removal.

### test_kill_no_selection_notifies
Pressing kill with no run selected (`selected_run_name = None`) must emit a "No run
selected" notification. Previously the action returned silently with no feedback,
leaving the user unable to tell whether their keypress was registered.

### test_submit_uses_table_cursor_not_stale_index
`_submit_selected` must submit whichever experiment the queue table cursor is on, not
the experiment at `selected_queue_idx`. The bug: `_refresh_queue_table` calls
`table.clear()`, which resets the cursor to row 0 and fires a `CursorMoved` event that
overwrites `selected_queue_idx` with 0 before `move_cursor` can restore it. Any
subsequent submit would always launch the first queue entry regardless of what was
visually selected.
