"""Shared fixtures and module-level patches for dashboard tests.

dashboard.py runs init_backends() and _load_project_config() at import time,
so we patch those before any test imports the module.
"""

import queue
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Minimal fakes for GPU / JobHandle
# ---------------------------------------------------------------------------

class FakeHandle:
    def __init__(self, run_name="test_run", gpu_id=0):
        self.run_name   = run_name
        self.gpu_id     = gpu_id
        self.log_file   = ""
        self.id         = "pid-fake"
        self.backend_id = "local"
        self._running   = True

    def is_running(self) -> bool:
        return self._running

    def cancel(self) -> None:
        self._running = False

    def open_log_reader(self) -> "queue.Queue":
        q: queue.Queue = queue.Queue()
        q.put(None)
        return q


class FakeGPU:
    index         = 0
    name          = "FakeGPU"
    cost_per_hour = 0.0
    backend_id    = "local"

    def status(self):
        from ripe_grm.compute_backend_client import GPUStatus
        return GPUStatus(util_pct=0, temp_c=40, mem_used_mb=100, mem_total_mb=8192)

    def submit(self, config):
        return FakeHandle(run_name=config.run_name, gpu_id=self.index)


FAKE_GPU = FakeGPU()


# ---------------------------------------------------------------------------
# Patch init_backends + project config before any import of dashboard
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def _patch_backends():
    """Patch init_backends and live_jobs for the entire test session."""
    backend_mock = MagicMock()
    backend_mock.init_backends.return_value = ([], {}, [FAKE_GPU], 1)
    backend_mock.live_jobs.return_value = {}
    sys.modules["ripe_grm.dashboard_backend"] = backend_mock
    yield
    sys.modules.pop("ripe_grm.dashboard_backend", None)
    # Also remove dashboard so it gets re-imported fresh if needed
    for mod in list(sys.modules):
        if "ripe_grm.dashboard" in mod:
            sys.modules.pop(mod, None)


@pytest.fixture
def default_params_file(tmp_path, monkeypatch):
    """Write a minimal default_params.json and point Path.cwd() at tmp_path."""
    import json
    params = {
        "n_envs": 1024,
        "n_epochs": 2,
        "learning_rate": 3e-4,
        "total_steps": 1_000_000,
        "hidden_size": 64,
        "vx_max": 3.0,
        "healthy_reward": 5.0,
        "forward_reward_weight": 1.0,
        "high_vel_cost_weight": 0.5,
        "ctrl_cost_weight": 0.1,
        "healthy_z_min": 1.0,
        "healthy_z_max": 2.0,
        "ent_coef": 0.01,
    }
    (tmp_path / "default_params.json").write_text(json.dumps(params))
    monkeypatch.chdir(tmp_path)
    return tmp_path
