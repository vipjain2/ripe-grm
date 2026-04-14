"""Vast.ai cloud GPU backend server.

Each "GPU slot" maps to one Vast.ai instance. Slots are idle until a job is
submitted; existing running instances are auto-attached at startup. All
shared server + SSH + poll infrastructure lives in compute_backend_server.py
— this module is a thin provider adapter.

Requires:
  - vastai SDK:  pip install vastai-sdk
  - API key:     VAST_API_KEY environment variable

compute_registry.json entry:
  {
    "sock_path":   "/tmp/drl_vast_backend.sock",
    "backend_id":  "vast",
    "script":      "vastai_backend.py",
    "num_slots":   2
  }

Run directly (normally started by the dashboard):
    python vastai_backend.py <sock_path> <backend_id>
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
from pathlib import Path

from vastai_sdk import VastAI

from ripe_grm.compute_backend_server import (
    InstanceSSH,
    _BaseCloudBackend,
    _BaseCloudGPUServer,
)
from ripe_grm.dashboard_log import log_error

DEFAULT_VAST_SOCK = "/tmp/drl_vast_backend.sock"
_REGISTRY_PATH = Path.cwd() / "compute_registry.json"


# ---------------------------------------------------------------------------
# Vast.ai SDK helpers
# ---------------------------------------------------------------------------

def _sdk() -> VastAI:
    """Return a thread-local VastAI SDK instance."""
    local = _sdk._local
    if not hasattr(local, "client"):
        api_key = os.environ.get("VAST_API_KEY", "")
        if not api_key:
            _key_file = Path.home() / ".config" / "vastai" / "vast_api_key"
            if _key_file.exists():
                api_key = _key_file.read_text().strip()
        local.client = VastAI(api_key=api_key)
    return local.client

_sdk._local = threading.local()


def _parse_response(result) -> list | dict | None:
    """Normalise SDK responses: return Python object regardless of str/dict/list."""
    if isinstance(result, (list, dict)):
        return result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception:
            return None
    return None


def _show_instances() -> list[dict]:
    """Return list of all instances."""
    try:
        result = _parse_response(_sdk().show_instances())
        return result if isinstance(result, list) else []
    except Exception as e:
        log_error("show_instances SDK call failed", exc=e)
        return []


def _get_instance(instance_id: str) -> dict | None:
    """Return instance info dict, or None if not found.

    Raises on SDK/network errors so callers can distinguish
    'instance confirmed gone' from 'could not check'.
    """
    result = _parse_response(_sdk().show_instance(id=int(instance_id)))
    if isinstance(result, dict) and result.get("id"):
        return result
    return None


def _instance_total_cost(instance_id: str) -> float:
    """Return total billed cost for an instance by summing its invoice line items."""
    try:
        items = _parse_response(_sdk().show_invoices())
        if not isinstance(items, list):
            return 0.0
        iid = int(instance_id)
        return sum(float(i.get("amount", 0)) for i in items
                   if i.get("instance_id") == iid)
    except Exception as e:
        log_error("instance_total_cost failed", exc=e, instance_id=instance_id)
        return 0.0


# ---------------------------------------------------------------------------
# _VastGPUServer
# ---------------------------------------------------------------------------

class _VastGPUServer(_BaseCloudGPUServer):
    """One virtual GPU slot backed by a Vast.ai instance."""

    _SLOT_NAME_PREFIX    = "Vast Slot"
    _SOCK_PATH_FMT       = "/tmp/drl_vast_{slot_id}.sock"
    _LOCAL_LOG_FMT       = "/tmp/drl_vast_{slot_id}.log"
    _REMOTE_PROJECT_DIR  = "/root/project"
    _REMOTE_FALLBACK_LOG = "/var/log/onstart.log"
    _LOG_PREFIX          = "[vast]"

    def _check_instance_alive(self, instance_id: str) -> dict | None:
        info = _get_instance(instance_id)
        if info and info.get("actual_status") == "running":
            return info
        return None

    def _build_ssh_from_info(self, info: dict) -> InstanceSSH:
        host = info.get("ssh_host") or info.get("public_ipaddr")
        port = int(info.get("ssh_port") or 22)
        return InstanceSSH(host, port)

    def _cost_from_info(self, info: dict) -> float:
        return float(info.get("dph_total", 0.0))

    def _total_cost_from_info(self, info: dict) -> float:
        iid = info.get("id")
        if iid is None:
            return 0.0
        return _instance_total_cost(str(iid))


# ---------------------------------------------------------------------------
# VastBackend
# ---------------------------------------------------------------------------

class VastBackend(_BaseCloudBackend):
    """Manages a pool of virtual GPU slots backed by Vast.ai instances."""

    _SLOT_CLASS = _VastGPUServer

    def _list_running_instances(self) -> list[dict]:
        return [i for i in _show_instances()
                if i.get("actual_status") == "running"]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _sock_path  = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VAST_SOCK
    _backend_id = sys.argv[2] if len(sys.argv) > 2 else "vast"

    # Read extra config from the matching registry entry
    _cfg: dict = {}
    if _REGISTRY_PATH.exists():
        for _entry in json.loads(_REGISTRY_PATH.read_text()):
            if _entry.get("sock_path") == _sock_path:
                _cfg = _entry
                break

    VastBackend(
        num_slots  = _cfg.get("num_slots", 1),
        sock_path  = _sock_path,
        backend_id = _backend_id,
    )
    print(f"Vast.ai backend ready  id={_backend_id}  sock={_sock_path}", flush=True)
    signal.pause()
