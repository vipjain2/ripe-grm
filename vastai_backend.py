"""Vast.ai cloud GPU backend server.

Each "GPU slot" maps to one Vast.ai instance. Slots are idle until a job is
submitted; submission rents the cheapest matching offer, syncs the project via
rsync, and starts training via SSH. Common server infrastructure lives in
compute_backend_server.py.

Requires:
  - vastai CLI:  pip install vastai
  - API key:     VAST_API_KEY environment variable

compute_registry.json entry:
  {
    "sock_path":   "/tmp/drl_vast_backend.sock",
    "backend_id":  "vast",
    "script":      "vastai_backend.py",
    "num_slots":   2,
    "offer_query": "num_gpus=1 gpu_name=RTX_4090 rentable=true verified=true",
    "image":       "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    "disk_gb":     20
  }

Run directly (normally started by the dashboard):
    python vastai_backend.py <sock_path> <backend_id>
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from ripe_autotrain.compute_backend_client import SocketJobHandle
from ripe_autotrain.compute_backend_server import _BaseBackend, _BackendServer, _BaseGPUServer

DEFAULT_VAST_SOCK = "/tmp/drl_vast_backend.sock"
_REGISTRY_PATH   = Path.cwd() / "compute_registry.json"

# SSH options applied to every connection
_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=10",
]


# ---------------------------------------------------------------------------
# Vast.ai CLI helpers
# ---------------------------------------------------------------------------

def _vastai(*args: str, check: bool = True) -> str:
    """Run a vastai CLI command and return stdout."""
    result = subprocess.run(
        ["vastai"] + list(args),
        capture_output=True, text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"vastai {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _get_instance(instance_id: str) -> dict | None:
    """Return instance info dict, or None if not found."""
    try:
        raw = _vastai("show", "instances", "--raw", check=False)
        instances = json.loads(raw) if raw else []
        return next((i for i in instances if str(i["id"]) == instance_id), None)
    except Exception:
        return None


def _wait_for_running(instance_id: str, timeout: int = 300) -> dict:
    """Poll until instance reaches 'running' status. Returns instance info."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = _get_instance(instance_id)
        if info and info.get("actual_status") == "running" and info.get("ssh_port"):
            return info
        time.sleep(5)
    raise TimeoutError(f"Instance {instance_id} did not start within {timeout}s")


def _ssh(host: str, port: int, cmd: str, timeout: int = 30) -> str:
    """Run a command on the remote instance, return stdout."""
    result = subprocess.run(
        ["ssh", f"-p{port}", *_SSH_OPTS, f"root@{host}", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def _wait_for_ssh(host: str, port: int, timeout: int = 120) -> None:
    """Poll until SSH accepts connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            _ssh(host, port, "echo ok", timeout=10)
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"SSH {host}:{port} not ready within {timeout}s")


# ---------------------------------------------------------------------------
# _VastGPUServer
# ---------------------------------------------------------------------------

class _VastGPUServer(_BaseGPUServer):
    """One virtual GPU slot backed by a Vast.ai instance.

    Idle when no instance is rented; active while an instance is running.
    """

    def __init__(self, slot_id: int, backend: "VastBackend"):
        super().__init__(slot_id, f"Vast Slot {slot_id}", f"/tmp/drl_vast_{slot_id}.sock")
        self._backend:        "VastBackend"  = backend
        self._instance_id:   str | None      = None
        self._ssh_host:      str | None      = None
        self._ssh_port:      int | None      = None
        self._remote_log:    str | None      = None
        self._cost_per_hour: float           = 0.0

    # -- Job-state helpers (called with self._lock held) ---------------------

    def _job_alive(self) -> bool:
        if self._instance_id is None:
            return False
        info = _get_instance(self._instance_id)
        return bool(info and info.get("actual_status") == "running")

    def _job_id(self) -> str:
        return str(self._instance_id) if self._instance_id else "-"

    def _on_job_ended(self) -> None:
        self._instance_id   = None
        self._ssh_host      = None
        self._ssh_port      = None
        self._remote_log    = None
        self._cost_per_hour = 0.0

    # -- Session restore -----------------------------------------------------

    def register_job(self, run_name: str, job_id: str,
                     log_file: str) -> "SocketJobHandle | None":
        """Re-attach to a Vast.ai instance that survived a dashboard restart."""
        info = _get_instance(job_id)
        if not info or info.get("actual_status") != "running":
            return None
        with self._lock:
            self._instance_id   = job_id
            self._ssh_host      = info.get("ssh_host") or info.get("public_ipaddr")
            self._ssh_port      = int(info.get("ssh_port") or 22)
            self._remote_log    = log_file
            self._run_name      = run_name
            self._log_file      = log_file
            self._cost_per_hour = float(info.get("dph_total", 0.0))
        return self._make_handle()

    # -- Override _cmd_info to report actual cost ----------------------------

    def _cmd_info(self, conn) -> None:
        with self._lock:
            cost = self._cost_per_hour
        conn.sendall((json.dumps({
            "index":         self._index,
            "name":          self.name,
            "cost_per_hour": cost,
            "sock_path":     self.sock_path,
        }) + "\n").encode())
        conn.close()

    # -- Hardware-specific commands -----------------------------------------

    def _cmd_status(self, conn) -> None:
        """GPU stats from the running instance, or zeroes if idle."""
        try:
            with self._lock:
                host = self._ssh_host
                port = self._ssh_port
            if host and port:
                out = _ssh(
                    host, port,
                    "nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,"
                    "memory.used,memory.total --format=csv,noheader,nounits",
                    timeout=10,
                )
                util, temp, mem_used, mem_total = [x.strip() for x in out.split(",")]
                conn.sendall((json.dumps({
                    "util_pct":    int(util),
                    "temp_c":      int(temp),
                    "mem_used_mb": int(mem_used),
                    "mem_total_mb": int(mem_total),
                }) + "\n").encode())
            else:
                # Idle slot — return zeroes
                conn.sendall((json.dumps({
                    "util_pct": 0, "temp_c": 0,
                    "mem_used_mb": 0, "mem_total_mb": 0,
                }) + "\n").encode())
        except Exception as e:
            conn.sendall((json.dumps({"error": str(e)}) + "\n").encode())
        finally:
            conn.close()

    def _cmd_submit(self, conn, json_str: str) -> None:
        try:
            cfg      = json.loads(json_str)
            run_name = cfg["run_name"]
            script   = cfg["script"]
            params   = cfg.get("params", {})
            chk      = cfg.get("checkpoint")
            log_name = Path(cfg.get("log_file", f"{run_name}.log")).name
            remote_log = f"/root/project/logs/{log_name}"

            # 1. Find cheapest matching offer
            raw    = _vastai("search", "offers", self._backend.offer_query, "--raw")
            offers = json.loads(raw)
            if not offers:
                raise RuntimeError(f"No offers match: {self._backend.offer_query!r}")
            offers.sort(key=lambda o: float(o.get("dph_total", 999)))
            offer_id = str(offers[0]["id"])

            # 2. Rent the instance
            result = json.loads(_vastai(
                "create", "instance", offer_id,
                "--image",  self._backend.image,
                "--disk",   str(self._backend.disk_gb),
                "--raw",
            ))
            instance_id = str(result.get("new_contract") or result["id"])

            # 3. Wait for the instance to be running and SSH-ready
            info     = _wait_for_running(instance_id)
            ssh_host = str(info.get("ssh_host") or info["public_ipaddr"])
            ssh_port = int(info.get("ssh_port") or 22)
            cost     = float(info.get("dph_total", 0.0))
            _wait_for_ssh(ssh_host, ssh_port)

            # 4. Sync project files to /root/project/
            subprocess.run([
                "rsync", "-az", "--delete",
                "-e", f"ssh -p {ssh_port} {' '.join(_SSH_OPTS)}",
                str(self._backend.project_root) + "/",
                f"root@{ssh_host}:/root/project/",
            ], check=True)

            # 5. Build training command and launch in the background
            cmd = (
                f"cd /root/project"
                f" && mkdir -p logs"
                f" && nohup {sys.executable} -u {script}"
                f" --run-name {run_name}"
            )
            for key, val in params.items():
                cmd += f" --{key.replace('_', '-')} {val}"
            if chk:
                cmd += f" --checkpoint {chk}"
            cmd += f" > {remote_log} 2>&1 &"
            _ssh(ssh_host, ssh_port, cmd)

            # 6. Update server state
            with self._lock:
                self._instance_id   = instance_id
                self._ssh_host      = ssh_host
                self._ssh_port      = ssh_port
                self._remote_log    = remote_log
                self._run_name      = run_name
                self._log_file      = remote_log
                self._cost_per_hour = cost

            self._backend._save_state()
            conn.sendall(f"ok {instance_id}\n".encode())
        except Exception as e:
            conn.sendall(f"error {e}\n".encode())
        finally:
            conn.close()

    def _cmd_cancel(self, conn) -> None:
        with self._lock:
            instance_id = self._instance_id
        try:
            if instance_id:
                _vastai("destroy", "instance", instance_id)
                conn.sendall(b"ok\n")
            else:
                conn.sendall(b"error no job running\n")
        except Exception as e:
            conn.sendall(f"error {e}\n".encode())
        finally:
            conn.close()

    def _cmd_logs(self, conn) -> None:
        with self._lock:
            host       = self._ssh_host
            port       = self._ssh_port
            remote_log = self._remote_log
        if not (host and port and remote_log):
            conn.close()
            return
        try:
            # Wait for the log file to appear on the remote instance
            deadline = time.time() + 60
            while time.time() < deadline:
                try:
                    _ssh(host, port, f"test -f {remote_log}", timeout=10)
                    break
                except Exception:
                    with self._lock:
                        if not self._job_alive():
                            return
                    time.sleep(2)

            # Stream via SSH tail -f; kill the tail process when the job ends
            proc = subprocess.Popen(
                ["ssh", f"-p{port}", *_SSH_OPTS, f"root@{host}",
                 f"tail -f {remote_log}"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            try:
                for line in proc.stdout:
                    conn.sendall(line)
                    with self._lock:
                        if not self._job_alive():
                            break
            finally:
                proc.terminate()
                proc.wait()
        except Exception:
            pass
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# VastBackend
# ---------------------------------------------------------------------------

class VastBackend(_BaseBackend):
    """Manages a pool of virtual GPU slots backed by Vast.ai instances.
    Pure server — use BackendClient from compute_backend_client.py to interact."""

    def __init__(self,
                 num_slots:    int        = 1,
                 offer_query:  str        = "num_gpus=1 rentable=true verified=true",
                 image:        str        = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
                 disk_gb:      int        = 20,
                 project_root: Path | None = None,
                 state_file:   Path | None = None,
                 sock_path:    str        = DEFAULT_VAST_SOCK,
                 backend_id:   str        = "vast"):
        self.offer_query  = offer_query
        self.image        = image
        self.disk_gb      = disk_gb
        self.project_root = project_root or Path.cwd()
        self._state_file  = state_file or (self.project_root / "vast_backend_state.json")
        self._servers: dict[int, _VastGPUServer] = {
            i: _VastGPUServer(i, self) for i in range(num_slots)
        }
        for server in self._servers.values():
            server.start()
        self._backend_server = _BackendServer(
            self._servers, self._save_state, sock_path, backend_id,
        )
        self._backend_server.start()
        self._restore_state()


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
        num_slots   = _cfg.get("num_slots",   1),
        offer_query = _cfg.get("offer_query", "num_gpus=1 rentable=true verified=true"),
        image       = _cfg.get("image",       "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"),
        disk_gb     = _cfg.get("disk_gb",     20),
        sock_path   = _sock_path,
        backend_id  = _backend_id,
    )
    print(f"Vast.ai backend ready  id={_backend_id}  sock={_sock_path}", flush=True)
    signal.pause()
