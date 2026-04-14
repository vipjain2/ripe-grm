"""Backend management — reads compute_registry.json, starts backend servers, and returns connected GPU clients."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from ripe_grm.compute_backend_client import BackendClient, GPU
from ripe_grm.dashboard_log import log_debug, log_error

_TRAINING_DIR    = Path.cwd()
BACKEND_REGISTRY = _TRAINING_DIR / "compute_registry.json"


def _read_registry() -> list[dict]:
    if not BACKEND_REGISTRY.exists():
        raise FileNotFoundError(f"Backend registry not found: {BACKEND_REGISTRY}")
    return json.loads(BACKEND_REGISTRY.read_text())


def _backend_responsive(sock_path: str) -> bool:
    try:
        BackendClient(sock_path).info()
        return True
    except Exception as e:
        log_debug("backend not responsive", sock_path=sock_path, error=str(e))
        return False


def _kill_existing_backend(script: str) -> None:
    """Kill any running processes whose cmdline contains the script name."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", script],
            capture_output=True, text=True,
        )
        pids = [int(p) for p in result.stdout.split() if p.strip()]
        for pid in pids:
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
        if pids:
            time.sleep(0.5)   # give processes a moment to exit
    except Exception as e:
        log_error("_kill_existing_backend failed", exc=e, script=script)


def _start_backend_servers() -> None:
    """Kill any existing backend processes and relaunch from the registry.
    Each backend type has its own script (local_backend.py, vast_backend.py, …).
    """
    base = Path(__file__).parent
    for entry in _read_registry():
        script = entry.get("script")
        if not script:
            continue
        _kill_existing_backend(script)
        subprocess.Popen(
            [sys.executable, str(base / script),
             entry["sock_path"],
             entry.get("backend_id", "local")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _connect_backends() -> list[BackendClient]:
    """Wait for each backend socket then return a BackendClient per entry."""
    clients = []
    for entry in _read_registry():
        sock_path = entry["sock_path"]
        deadline  = time.time() + 10.0
        while not _backend_responsive(sock_path):
            if time.time() > deadline:
                break
            time.sleep(0.1)
        clients.append(BackendClient(sock_path))
    return clients


def live_jobs(backends: list[BackendClient]) -> dict:
    """Return {run_name: handle} for all running jobs across every backend.

    Per-backend failures (socket errors, server exceptions) are logged and
    skipped so one flaky backend can't block dashboard startup.
    """
    live: dict = {}
    for b in backends:
        try:
            for h in b.running_jobs():
                live[h.run_name] = h
        except Exception as e:
            log_error("running_jobs failed", sock_path=b._sock_path, exc=e)
    return live


def init_backends() -> tuple[list[BackendClient], dict[str, BackendClient], list[GPU], int]:
    """Start backend servers, connect to them, and return runtime state.

    Returns:
        (backends, backend_by_id, gpus, n_gpus)
    """
    _start_backend_servers()
    backends      = _connect_backends()
    backend_infos = [b.info() for b in backends]
    backend_by_id = {
        info["backend_id"]: b for b, info in zip(backends, backend_infos)
    }
    gpus   = [g for b in backends for g in b.discover_gpus()]
    n_gpus = len(gpus)
    return backends, backend_by_id, gpus, n_gpus
