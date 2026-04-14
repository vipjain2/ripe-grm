"""RunPod cloud GPU backend server.

Each "GPU slot" maps to one RunPod pod. Slots are idle until a job is
submitted or an existing pod is discovered. All shared server + SSH + poll
infrastructure lives in compute_backend_server.py — this module is a thin
provider adapter.

Requires:
  - runpod SDK:  pip install runpod
  - API key:     RUNPOD_API_KEY environment variable
  - SSH key added to RunPod account settings

compute_registry.json entry:
  {
    "sock_path":   "/tmp/drl_runpod_backend.sock",
    "backend_id":  "runpod",
    "script":      "runpod_backend.py",
    "num_slots":   2
  }

Run directly (normally started by the dashboard):
    python runpod_backend.py <sock_path> <backend_id>
"""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path

import runpod

from ripe_grm.compute_backend_server import (
    InstanceSSH,
    _BaseCloudBackend,
    _BaseCloudGPUServer,
)
from ripe_grm.dashboard_log import log_error

DEFAULT_RUNPOD_SOCK = "/tmp/drl_runpod_backend.sock"
_REGISTRY_PATH = Path.cwd() / "compute_registry.json"


# ---------------------------------------------------------------------------
# RunPod SDK helpers
# ---------------------------------------------------------------------------

def _init_api_key() -> None:
    """Set the runpod API key from env or config file."""
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        key_file = Path.home() / ".config" / "runpod" / "api_key"
        if key_file.exists():
            api_key = key_file.read_text().strip()
    if api_key:
        runpod.api_key = api_key


def _get_pods() -> list[dict]:
    """Return list of all pods."""
    try:
        return runpod.get_pods() or []
    except Exception as e:
        log_error("get_pods failed", exc=e)
        return []


def _get_pod(pod_id: str) -> dict | None:
    """Return pod info dict, or None if not found.

    Raises on SDK/network errors so callers can distinguish
    'pod confirmed gone' from 'could not check'.
    """
    pod = runpod.get_pod(pod_id)
    if isinstance(pod, dict) and pod.get("id"):
        return pod
    return None


def _ssh_from_pod(pod: dict) -> InstanceSSH:
    """Build InstanceSSH from a RunPod pod info dict.

    RunPod exposes SSH via runtime.ports — find the entry mapping
    privatePort=22 to a public IP/port. Raises if the pod has no SSH
    endpoint (runtime not yet populated).
    """
    runtime = pod.get("runtime") or {}
    for p in runtime.get("ports") or []:
        if p.get("privatePort") == 22 and p.get("isIpPublic"):
            return InstanceSSH(p["ip"], int(p["publicPort"]))
    # Fallback: top-level publicIp with portMappings (older SDK shape)
    host = pod.get("publicIp")
    if host:
        mappings = pod.get("portMappings") or {}
        return InstanceSSH(host, int(mappings.get("22", 22)))
    raise RuntimeError(f"Pod {pod.get('id')} has no SSH endpoint (runtime not ready?)")


# ---------------------------------------------------------------------------
# _RunPodGPUServer
# ---------------------------------------------------------------------------

class _RunPodGPUServer(_BaseCloudGPUServer):
    """One virtual GPU slot backed by a RunPod pod."""

    _SLOT_NAME_PREFIX    = "RunPod Slot"
    _SOCK_PATH_FMT       = "/tmp/drl_runpod_{slot_id}.sock"
    _LOCAL_LOG_FMT       = "/tmp/drl_runpod_{slot_id}.log"
    _REMOTE_PROJECT_DIR  = "/workspace/project"
    _REMOTE_FALLBACK_LOG = "/var/log/syslog"
    _LOG_PREFIX          = "[runpod]"

    def _check_instance_alive(self, instance_id: str) -> dict | None:
        pod = _get_pod(instance_id)
        if pod and pod.get("desiredStatus") == "RUNNING":
            return pod
        return None

    def _build_ssh_from_info(self, info: dict) -> InstanceSSH:
        return _ssh_from_pod(info)

    def _cost_from_info(self, info: dict) -> float:
        return float(info.get("costPerHr", 0.0))


# ---------------------------------------------------------------------------
# RunPodBackend
# ---------------------------------------------------------------------------

class RunPodBackend(_BaseCloudBackend):
    """Manages a pool of virtual GPU slots backed by RunPod pods."""

    _SLOT_CLASS = _RunPodGPUServer

    def _pre_init(self) -> None:
        _init_api_key()

    def _list_running_instances(self) -> list[dict]:
        # desiredStatus is the target — runtime is populated only once the pod
        # is actually started and SSH is reachable. Skip provisioning pods.
        return [p for p in _get_pods()
                if p.get("desiredStatus") == "RUNNING" and p.get("runtime")]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _sock_path  = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUNPOD_SOCK
    _backend_id = sys.argv[2] if len(sys.argv) > 2 else "runpod"

    _cfg: dict = {}
    if _REGISTRY_PATH.exists():
        for _entry in json.loads(_REGISTRY_PATH.read_text()):
            if _entry.get("sock_path") == _sock_path:
                _cfg = _entry
                break

    RunPodBackend(
        num_slots  = _cfg.get("num_slots", 1),
        sock_path  = _sock_path,
        backend_id = _backend_id,
    )
    print(f"RunPod backend ready  id={_backend_id}  sock={_sock_path}", flush=True)
    signal.pause()
