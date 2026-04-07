"""Vast.ai cloud GPU backend server.

Each "GPU slot" maps to one Vast.ai instance. Slots are idle until a job is
submitted; submission rents the cheapest matching offer, syncs the project via
rsync, and starts training via SSH. Common server infrastructure lives in
compute_backend_server.py.

Requires:
  - vastai SDK:  pip install vastai
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
import threading
import time
from pathlib import Path

from vastai import VastAI

from ripe_autotrain.compute_backend_client import SocketJobHandle
from ripe_autotrain.compute_backend_server import _BackendServer, _BaseGPUServer

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
# Vast.ai SDK helpers
# ---------------------------------------------------------------------------

def _sdk() -> VastAI:
    """Return a thread-local VastAI SDK instance."""
    local = _sdk._local
    if not hasattr(local, "client"):
        api_key = os.environ.get("VAST_API_KEY", "")
        local.client = VastAI(api_key=api_key, raw=True)
    return local.client

_sdk._local = threading.local()


def _show_instances() -> list[dict]:
    """Return list of all instances. SDK raw=True returns list[dict] directly."""
    try:
        result = _sdk().show_instances(quiet=False)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def _get_instance(instance_id: str) -> dict | None:
    """Return instance info dict, or None if not found."""
    try:
        result = _sdk().show_instance(id=int(instance_id))
        if isinstance(result, dict) and result.get("id"):
            return result
        return None
    except Exception:
        return None


def _wait_for_running(instance_id: str, timeout: int = 300) -> dict:
    """Poll until instance reaches 'running' status. Returns instance info."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        instances = _show_instances()
        info = next((i for i in instances if str(i["id"]) == instance_id), None)
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
    The background poll thread (started by _BaseGPUServer.start) handles all
    SSH work: alive checks, log fetching, and nvidia-smi updates.
    """

    _POLL_INTERVAL    = 10   # seconds between poll iterations
    _STATUS_MIN_SECS  = 30   # minimum seconds between nvidia-smi calls

    def __init__(self, slot_id: int, backend: "VastBackend"):
        super().__init__(slot_id, f"Vast Slot {slot_id}", f"/tmp/drl_vast_{slot_id}.sock")
        self._backend:        "VastBackend"  = backend
        self._instance_id:   str | None      = None
        self._ssh_host:      str | None      = None
        self._ssh_port:      int | None      = None
        self._remote_log:    str | None      = None
        self._cost_per_hour: float           = 0.0
        self._total_cost:    float           = 0.0
        self._instance_alive: bool           = False
        self._absent_polls:  int             = 0
        self._training_alive: bool           = False  # training process running on instance
        self._absent_training: int           = 0      # consecutive polls with no training process
        # Local log file — poll thread downloads SSH output here; _cmd_logs reads it
        self._local_log:     str             = f"/tmp/drl_vast_{slot_id}.log"
        self._log_offset:    int             = 0   # lines fetched from remote so far
        self._log_complete:  bool            = False  # True after final download done
        # GPU status cache — poll thread writes; _cmd_status reads
        self._status_cache:  dict            = {"util_pct": 0, "temp_c": 0,
                                                 "mem_used_mb": 0, "mem_total_mb": 0}
        self._status_lock    = threading.Lock()
        self._last_status_t: float           = 0.0

    # -- Job-state helpers (called with self._lock held) ---------------------

    def _job_alive(self) -> bool:
        return self._instance_alive and self._training_alive

    def _job_id(self) -> str:
        return str(self._instance_id) if self._instance_id else "-"

    def _on_job_ended(self) -> None:
        self._instance_id    = None
        self._ssh_host       = None
        self._ssh_port       = None
        self._remote_log     = None
        self._cost_per_hour  = 0.0
        self._instance_alive  = False
        self._training_alive  = False
        self._absent_training = 0
        self._log_offset      = 0
        self._log_complete    = True   # stop any waiting _cmd_logs
        with self._status_lock:
            self._status_cache = {"util_pct": 0, "temp_c": 0,
                                  "mem_used_mb": 0, "mem_total_mb": 0}

    # -- Session restore -----------------------------------------------------

    def register_job(self, run_name: str, job_id: str,
                     log_file: str) -> "SocketJobHandle | None":
        """Re-attach to a Vast.ai instance that survived a dashboard restart."""
        info = _get_instance(job_id)
        if not info or info.get("actual_status") != "running":
            return None
        ssh_host = info.get("ssh_host") or info.get("public_ipaddr")
        ssh_port = int(info.get("ssh_port") or 22)
        # Check training process is actually running — instance alive != job alive
        training_alive = False
        try:
            out = _ssh(ssh_host, ssh_port,
                       f"pgrep -fa 'train.py' | grep -- '--run-name {run_name}'",
                       timeout=15)
            training_alive = bool(out.strip())
        except Exception:
            training_alive = True  # SSH failed; assume alive, poll will correct
        if not training_alive:
            return None
        with self._lock:
            self._instance_id    = job_id
            self._ssh_host       = ssh_host
            self._ssh_port       = ssh_port
            self._remote_log     = log_file
            self._run_name       = run_name
            self._log_file       = self._local_log
            self._cost_per_hour  = float(info.get("dph_total", 0.0))
            self._instance_alive = True
            self._absent_polls   = 0
            self._training_alive  = True
            self._absent_training = 0
            self._log_offset     = 0
            self._log_complete   = False
        return self._make_handle()

    # -- Discover extra fields -----------------------------------------------

    def _discover_extra(self) -> dict:
        with self._lock:
            return {
                "cost_per_hour": self._cost_per_hour,
                "instance_id":   self._instance_id,
                "total_cost":    self._total_cost,
            }

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

    # -- Background poll (runs in poll thread, no blocking calls elsewhere) --

    def _poll(self) -> None:
        """Single poll iteration:
        1. Alive check via API
        2. Discover log path if unknown
        3. Fetch new log lines
        4. Final log download when instance just died
        5. nvidia-smi (throttled to _STATUS_MIN_SECS)
        """
        with self._lock:
            instance_id      = self._instance_id
            ssh_host         = self._ssh_host
            ssh_port         = self._ssh_port
            remote_log       = self._remote_log
            run_name         = self._run_name or ""
            was_inst_alive   = self._instance_alive
            was_train_alive  = self._training_alive

        if not instance_id:
            return

        # 1. Instance alive check via API
        info     = _get_instance(instance_id)
        is_alive = bool(info and info.get("actual_status") == "running")

        with self._lock:
            if self._instance_id == instance_id:
                if is_alive:
                    self._absent_polls   = 0
                    self._instance_alive = True
                    if info:
                        dph      = float(info.get("dph_total", self._cost_per_hour))
                        uptime   = float(info.get("uptime_mins", 0.0))
                        self._total_cost = dph * uptime / 60.0
                else:
                    self._absent_polls += 1
                    if self._absent_polls >= 2:
                        self._instance_alive = False

        if not (ssh_host and ssh_port) or not is_alive:
            return

        # 2. Training process alive check via pgrep
        try:
            out = _ssh(ssh_host, ssh_port,
                       f"pgrep -fa 'train.py' | grep -- '--run-name {run_name}'",
                       timeout=10)
            process_found = bool(out.strip())
        except Exception:
            process_found = True  # SSH error; keep current state, poll will retry

        with self._lock:
            if self._instance_id == instance_id:
                if process_found:
                    self._absent_training = 0
                    self._training_alive  = True
                else:
                    self._absent_training += 1
                    if self._absent_training >= 2:
                        self._training_alive = False

        is_training_alive = process_found or self._absent_training < 2

        # 3. Discover remote log path when not yet known
        if not remote_log and run_name:
            try:
                found = _ssh(ssh_host, ssh_port,
                             f"ls -t /root/project/logs/{run_name}*.log 2>/dev/null | head -1",
                             timeout=10).strip()
                remote_log = found or "/var/log/onstart.log"
                with self._lock:
                    if self._instance_id == instance_id:
                        self._remote_log = remote_log
            except Exception:
                pass

        if not remote_log:
            return

        # 4. Fetch new log lines while training is alive
        if is_training_alive:
            self._fetch_log_lines(ssh_host, ssh_port, remote_log)

        # 5. Final download when training just confirmed ended; signal log complete
        training_just_ended = was_train_alive and not is_training_alive
        if training_just_ended:
            self._fetch_log_lines(ssh_host, ssh_port, remote_log, final=True)
            self._log_complete = True

        # 6. Instance died: final download + log complete
        inst_just_died = was_inst_alive and not is_alive and self._absent_polls >= 2
        if inst_just_died and not self._log_complete:
            self._fetch_log_lines(ssh_host, ssh_port, remote_log, final=True)
            self._log_complete = True

        # 7. nvidia-smi — only when instance alive and enough time has passed
        if time.time() - self._last_status_t >= self._STATUS_MIN_SECS:
            self._refresh_status_cache(ssh_host, ssh_port)
            self._last_status_t = time.time()

    def _fetch_log_lines(self, host: str, port: int, remote_log: str,
                         final: bool = False) -> None:
        """SSH tail to fetch lines since _log_offset; append to local log file."""
        timeout = 30 if final else 10
        try:
            out = _ssh(host, port,
                       f"tail -n +{self._log_offset + 1} {remote_log} 2>/dev/null",
                       timeout=timeout)
            if out:
                new_lines = out.splitlines()
                with open(self._local_log, "a") as f:
                    for line in new_lines:
                        f.write(line + "\n")
                self._log_offset += len(new_lines)
        except Exception:
            pass

    def _refresh_status_cache(self, host: str, port: int) -> None:
        """Fetch nvidia-smi from the instance and update the cached status."""
        try:
            out = _ssh(
                host, port,
                "nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,"
                "memory.used,memory.total --format=csv,noheader,nounits",
                timeout=12,
            )
            util, temp, mem_used, mem_total = [x.strip() for x in out.split(",")]
            with self._status_lock:
                self._status_cache = {
                    "util_pct":    int(util),
                    "temp_c":      int(temp),
                    "mem_used_mb": int(mem_used),
                    "mem_total_mb": int(mem_total),
                }
        except Exception:
            pass

    # -- Hardware-specific commands -----------------------------------------

    def _cmd_status(self, conn) -> None:
        """Return cached GPU stats — no blocking SSH call."""
        try:
            with self._status_lock:
                payload = dict(self._status_cache)
            conn.sendall((json.dumps(payload) + "\n").encode())
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

            with self._lock:
                ssh_host    = self._ssh_host
                ssh_port    = self._ssh_port
                instance_id = self._instance_id

            if not (ssh_host and ssh_port and instance_id):
                raise RuntimeError("No instance attached to this slot — attach one first.")

            # 1. Sync project files to /root/project/
            subprocess.run([
                "rsync", "-az", "--delete",
                "-e", f"ssh -p {ssh_port} {' '.join(_SSH_OPTS)}",
                str(self._backend.project_root) + "/",
                f"root@{ssh_host}:/root/project/",
            ], check=True)

            # 2. Build training command and launch in the background
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

            # 3. Update server state (poll thread will start fetching logs/status)
            Path(self._local_log).unlink(missing_ok=True)  # fresh file for new job
            with self._lock:
                self._remote_log     = remote_log
                self._run_name       = run_name
                self._log_file       = self._local_log
                self._training_alive  = True
                self._absent_training = 0
                self._log_offset     = 0
                self._log_complete   = False

            conn.sendall(f"ok {instance_id}\n".encode())
        except Exception as e:
            conn.sendall(f"error {e}\n".encode())
        finally:
            conn.close()

    def _cmd_cancel(self, conn) -> None:
        with self._lock:
            instance_id = self._instance_id
            ssh_host    = self._ssh_host
            ssh_port    = self._ssh_port
            run_name    = self._run_name
        try:
            if not instance_id:
                conn.sendall(b"error no job running\n")
                return
            if not (ssh_host and ssh_port):
                conn.sendall(b"error no ssh connection to instance\n")
                return
            # Kill only the training process — leave the instance running
            _ssh(ssh_host, ssh_port,
                 f"pkill -f 'train.py.*--run-name {run_name}'",
                 timeout=15)
            conn.sendall(b"ok\n")
        except Exception as e:
            conn.sendall(f"error {e}\n".encode())
        finally:
            conn.close()

    def _cmd_logs(self, conn) -> None:
        """Stream the local log file written by the poll thread. No SSH."""
        local_log = self._local_log
        try:
            # Wait for the file to appear (poll thread creates it on first fetch)
            while not Path(local_log).exists():
                if self._log_complete:
                    return
                time.sleep(0.5)
            with open(local_log) as f:
                while True:
                    line = f.readline()
                    if line:
                        conn.sendall(line.encode())
                    else:
                        if self._log_complete:
                            break
                        time.sleep(0.5)
        except Exception:
            pass
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# VastBackend
# ---------------------------------------------------------------------------

class VastBackend:
    """Manages a pool of virtual GPU slots backed by Vast.ai instances.
    Pure server — use BackendClient from compute_backend_client.py to interact."""

    def __init__(self,
                 num_slots:    int        = 1,
                 offer_query:  str        = "num_gpus=1 rentable=true verified=true",
                 image:        str        = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
                 disk_gb:      int        = 20,
                 project_root: Path | None = None,
                 sock_path:    str        = DEFAULT_VAST_SOCK,
                 backend_id:   str        = "vast"):
        self.offer_query  = offer_query
        self.image        = image
        self.disk_gb      = disk_gb
        self.project_root = project_root or Path.cwd()
        self._servers: dict[int, _VastGPUServer] = {
            i: _VastGPUServer(i, self) for i in range(num_slots)
        }
        for server in self._servers.values():
            server.start()
        self._backend_server = _BackendServer(
            self._servers, sock_path, backend_id,
        )
        self._backend_server.start()
        threading.Thread(target=self._poll_instances_loop, daemon=True,
                         name="vast-instance-poll").start()

    def _poll_instances_loop(self) -> None:
        """Background thread: discover running Vast.ai instances every 30s
        and attach any untracked ones to idle slots."""
        try:
            self._attach_untracked_instances()
        except Exception:
            pass
        while True:
            time.sleep(30)
            try:
                self._attach_untracked_instances()
            except Exception:
                pass

    def _attach_untracked_instances(self) -> None:
        """Discover running Vast.ai instances not yet tracked and attach them to
        idle slots. Alive checking and log/status fetching are handled by each
        slot's _poll() thread."""
        instances = _show_instances()
        running = [
            i for i in instances
            if i.get("actual_status") == "running" and i.get("ssh_port")
        ]

        tracked = set()
        for server in self._servers.values():
            with server._lock:
                if server._instance_id:
                    tracked.add(str(server._instance_id))

        for info in running:
            if str(info["id"]) in tracked:
                continue

            # Attach to an idle slot, or create a new one
            idle = next(
                (s for s in self._servers.values() if s._instance_id is None),
                None,
            )
            if idle is None:
                slot_id = max(self._servers) + 1
                idle = _VastGPUServer(slot_id, self)
                idle.start()
                self._servers[slot_id] = idle

            ssh_host = info.get("ssh_host") or info.get("public_ipaddr")
            ssh_port = int(info.get("ssh_port") or 22)

            # Discover run name from the running training process
            run_name = info.get("label") or f"instance-{info['id']}"
            training_found = False
            try:
                ps_out = _ssh(ssh_host, ssh_port,
                              "ps -eo args | grep 'train.py' | grep -v grep | head -1",
                              timeout=10)
                parts = ps_out.split()
                training_found = bool(parts)
                if "--run-name" in parts:
                    run_name = parts[parts.index("--run-name") + 1]
            except Exception:
                continue  # SSH failed during discovery — skip and retry next cycle

            if not training_found:
                continue  # instance alive but no training process — skip

            with idle._lock:
                idle._instance_id     = str(info["id"])
                idle._ssh_host        = ssh_host
                idle._ssh_port        = ssh_port
                idle._cost_per_hour   = float(info.get("dph_total", 0.0))
                idle._run_name        = run_name
                idle._log_file        = idle._local_log
                idle._remote_log      = ""   # poll thread will discover remote path
                idle._instance_alive  = True
                idle._absent_polls    = 0
                idle._training_alive  = True
                idle._absent_training = 0
                idle._log_offset      = 0
                idle._log_complete    = False


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
