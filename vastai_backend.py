"""Vast.ai cloud GPU backend server.

Each "GPU slot" maps to one Vast.ai instance. Slots are idle until a job is
submitted; submission rents the cheapest matching offer, syncs the project via
rsync, and starts training via SSH. Common server infrastructure lives in
compute_backend_server.py.

Requires:
  - vastai SDK:  pip install vastai-sdk
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
from ripe_autotrain.dashboard_log import log_debug, log_error
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from vastai_sdk import VastAI

from ripe_autotrain.compute_backend_client import SocketJobHandle
from ripe_autotrain.compute_backend_server import _BackendServer, _BaseGPUServer

DEFAULT_VAST_SOCK = "/tmp/drl_vast_backend.sock"
_REGISTRY_PATH   = Path.cwd() / "compute_registry.json"



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
    """Return instance info dict, or None if not found."""
    try:
        result = _parse_response(_sdk().show_instance(id=int(instance_id)))
        if isinstance(result, dict) and result.get("id"):
            return result
        return None
    except Exception as e:
        log_error("get_instance SDK call failed", exc=e, instance_id=instance_id)
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


def _ssh_execute_async(ssh: "InstanceSSH", cmd: str, instance_id: str = "") -> None:
    """Fire-and-forget command via SSH in a background thread."""
    def _run() -> None:
        try:
            out = ssh.run(cmd, timeout=30)
            if out.strip():
                log_debug("ssh execute output", instance_id=instance_id,
                          cmd=cmd[:80], output=out[:500])
        except Exception as e:
            log_error("ssh execute failed", exc=e, instance_id=instance_id,
                      cmd=cmd[:80])
    threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# InstanceSSH — SSH wrapper for monitoring commands that need output
# ---------------------------------------------------------------------------

_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=10",
]


class InstanceSSH:
    """SSH connection to a single Vast.ai instance.

    Used for all remote commands: monitoring (pgrep, tail, nvidia-smi, ps),
    launch and cancel (_ssh_execute_async), and file transfer (copy_to/SCP).
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def run(self, cmd: str, timeout: int = 30) -> str:
        """Run a command and return stdout. Raises RuntimeError on non-zero exit."""
        result = subprocess.run(
            ["ssh", f"-p{self.port}", *_SSH_OPTS, f"root@{self.host}", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        return result.stdout.strip()

    def copy_to(self, local_path: str, remote_path: str, timeout: int = 60) -> None:
        """Copy a local file to the instance via SCP."""
        result = subprocess.run(
            ["scp", f"-P{self.port}", *_SSH_OPTS, local_path,
             f"root@{self.host}:{remote_path}"],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

    def copy_from(self, remote_path: str, local_path: str, timeout: int = 120) -> None:
        """Copy a file from the instance to the local machine via SCP."""
        result = subprocess.run(
            ["scp", f"-P{self.port}", *_SSH_OPTS,
             f"root@{self.host}:{remote_path}", local_path],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

    def wait_ready(self, timeout: int = 120) -> None:
        """Block until SSH accepts connections."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self.run("echo ok", timeout=10)
                return
            except Exception:
                time.sleep(5)
        raise TimeoutError(f"SSH {self.host}:{self.port} not ready within {timeout}s")

    @classmethod
    def from_instance(cls, info: dict) -> "InstanceSSH":
        """Build from a Vast.ai instance info dict."""
        host = info.get("ssh_host") or info.get("public_ipaddr")
        port = int(info.get("ssh_port") or 22)
        return cls(host, port)


def _wait_for_running(instance_id: str, timeout: int = 300) -> dict:
    """Poll until instance reaches 'running' status. Returns instance info."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        instances = _show_instances()
        info = next((i for i in instances if str(i["id"]) == instance_id), None)
        if info and info.get("actual_status") == "running":
            return info
        time.sleep(5)
    raise TimeoutError(f"Instance {instance_id} did not start within {timeout}s")


def _wait_for_execute(instance_id: str, timeout: int = 120) -> None:
    """Poll until execute API accepts commands on the instance."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            out = _execute(instance_id, "echo ok", timeout=10)
            if "ok" in out:
                return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"Instance {instance_id} not ready for commands within {timeout}s")



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

    @property
    def python_cmd(self) -> list[str]:
        return ["uv", "run", "--project", "/app", "python", "-u"]

    def __init__(self, slot_id: int, backend: "VastBackend"):
        super().__init__(slot_id, f"Vast Slot {slot_id}", f"/tmp/drl_vast_{slot_id}.sock")
        self._backend:        "VastBackend"  = backend
        self._instance_id:   str | None      = None
        self._ssh:           InstanceSSH | None = None
        self._remote_log:    str | None      = None
        self._cost_per_hour: float           = 0.0
        self._total_cost:    float           = 0.0
        self._instance_alive: bool           = False
        self._absent_polls:  int             = 0
        # Training state machine: "idle" | "submitted" | "running"
        self._training_status: str           = "idle"
        self._submitted_at:   float          = 0.0
        self._absent_training: int           = 0  # consecutive missing pgrep hits (in "running")
        self._ssh_fail_polls:  int           = 0  # consecutive SSH failures in pgrep check
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
        return self._instance_alive and self._training_status in ("submitted", "running", "downloading")

    def _job_id(self) -> str:
        return str(self._instance_id) if self._instance_id else "-"

    def _on_job_ended(self) -> None:
        """Clear training state. Instance may still be alive."""
        self._training_status = "idle"
        self._submitted_at    = 0.0
        self._absent_training = 0
        self._ssh_fail_polls  = 0
        self._log_offset      = 0
        self._log_complete    = True
        self._remote_log      = None

    def _on_instance_detached(self) -> None:
        """Clear all instance state — slot becomes fully idle."""
        self._on_job_ended()
        self._instance_id    = None
        self._ssh            = None
        self._cost_per_hour  = 0.0
        self._instance_alive = False
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
        ssh = InstanceSSH.from_instance(info)
        # Check training process is actually running — instance alive != job alive
        training_alive = False
        try:
            out = ssh.run(f"pgrep -fa 'train.py' | grep -- '--run-name {run_name}' || true",
                          timeout=15)
            training_alive = bool(out.strip())
        except Exception:
            training_alive = True  # SSH failed; assume alive, poll will correct
        if not training_alive:
            return None
        with self._lock:
            self._instance_id    = job_id
            self._ssh            = ssh
            self._remote_log     = log_file
            self._run_name       = run_name
            self._log_file       = self._local_log
            self._cost_per_hour  = float(info.get("dph_total", 0.0))
            self._instance_alive  = True
            self._absent_polls    = 0
            self._training_status = "running"
            self._submitted_at    = 0.0
            self._absent_training = 0
            self._ssh_fail_polls  = 0
            self._log_offset      = 0
            self._log_complete    = False
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
            ssh              = self._ssh
            remote_log       = self._remote_log
            run_name         = self._run_name or ""
            was_inst_alive   = self._instance_alive
            training_status  = self._training_status

        if not instance_id:
            return

        # 1. Instance alive check via API
        info     = _get_instance(instance_id)
        is_alive = bool(info and info.get("actual_status") == "running")

        inst_gone = False
        with self._lock:
            if self._instance_id == instance_id:
                if is_alive:
                    self._absent_polls   = 0
                    self._instance_alive = True
                    if info:
                        self._cost_per_hour = float(info.get("dph_total", self._cost_per_hour))
                        self._total_cost    = _instance_total_cost(instance_id)
                else:
                    self._absent_polls += 1
                    inst_gone = self._absent_polls >= 2

        if not is_alive:
            # Instance disappeared — attempt final log/checkpoint download, then detach
            if inst_gone and not self._log_complete:
                try:
                    self._fetch_log_lines(ssh, remote_log, final=True)
                    self._download_checkpoints(ssh, run_name, instance_id)
                except Exception:
                    pass  # instance is gone, best-effort
                self._log_complete = True
            if inst_gone:
                with self._lock:
                    if self._instance_id == instance_id:
                        self._on_instance_detached()
            return

        # 2. Training process state machine — pgrep only when not idle
        is_training_alive = False
        if training_status in ("submitted", "running"):
            # Check for the specifically staged run_name
            try:
                out = ssh.run(
                    f"pgrep -fa 'train.py' | grep -E -- '--run-name {run_name}( |$)' || true",
                    timeout=10)
                process_found: bool | None = bool(out.strip())
                with self._lock:
                    if self._instance_id == instance_id:
                        self._ssh_fail_polls = 0
            except Exception as e:
                with self._lock:
                    if self._instance_id == instance_id:
                        self._ssh_fail_polls += 1
                        ssh_fails = self._ssh_fail_polls
                if ssh_fails >= 5:
                    # 5 consecutive SSH failures (~50s) — treat as process not found
                    log_error("SSH pgrep failed 5x in a row; assuming training ended",
                              instance_id=instance_id, run_name=run_name, error=str(e))
                    process_found = False
                else:
                    log_debug("pgrep SSH check failed; retaining current state",
                              instance_id=instance_id, run_name=run_name,
                              fail_count=ssh_fails, error=str(e))
                    process_found = None  # unknown — keep current state

            # If staged run not found, do a broad scan for any train.py process
            detected_name: str | None = None
            if process_found is False:
                try:
                    broad_out = ssh.run(
                        "ps -eo args | grep 'train.py' | grep -v grep | head -1",
                        timeout=10)
                    if broad_out.strip():
                        parts = broad_out.strip().split()
                        if "--run-name" in parts:
                            detected_name = parts[parts.index("--run-name") + 1]
                except Exception as e:
                    log_debug("Broad process scan failed", instance_id=instance_id,
                              error=str(e))

            with self._lock:
                if self._instance_id == instance_id:
                    if process_found:
                        self._training_status = "running"
                        self._absent_training = 0
                    elif detected_name:
                        # A different training process is running — switch tracking to it
                        log_debug("Switching tracked run to detected process",
                                  was=run_name, now=detected_name, instance_id=instance_id)
                        Path(self._local_log).unlink(missing_ok=True)
                        self._run_name        = detected_name
                        self._log_file        = self._local_log
                        self._remote_log      = ""   # log discovery will find the real path
                        self._training_status = "running"
                        self._absent_training = 0
                        self._log_offset      = 0
                        self._log_complete    = False
                    elif process_found is False:
                        # No training process at all — apply timeout/absent logic
                        if training_status == "submitted":
                            if time.time() - self._submitted_at > 60.0:
                                log_error("Training process never confirmed within 60s",
                                          instance_id=instance_id, run_name=run_name)
                                self._training_status = "idle"
                                self._run_name        = None
                                self._log_file        = ""
                        elif training_status == "running":
                            self._absent_training += 1
                            if self._absent_training >= 2:
                                self._training_status = "downloading"
                    is_training_alive = self._training_status in ("submitted", "running")

        # 3. Discover remote log path when not yet known
        if not remote_log and run_name:
            try:
                found = ssh.run(
                    f"ls -t /root/project/logs/{run_name}*.log 2>/dev/null | head -1",
                    timeout=10).strip()
                remote_log = found or "/var/log/onstart.log"
                with self._lock:
                    if self._instance_id == instance_id:
                        self._remote_log = remote_log
            except Exception as e:
                log_debug("remote log discovery failed", instance_id=instance_id,
                          run_name=run_name, error=str(e))

        if not remote_log:
            return

        # 4. Fetch new log lines while training is alive
        if is_training_alive:
            self._fetch_log_lines(ssh, remote_log)

        # 5. Download checkpoints + final log when training process ended
        is_downloading = training_status == "downloading" or (
            training_status in ("submitted", "running") and not is_training_alive
        )
        if is_downloading and not self._log_complete:
            self._fetch_log_lines(ssh, remote_log, final=True)
            self._download_checkpoints(ssh, run_name, instance_id)
            with self._lock:
                if self._instance_id == instance_id:
                    self._on_job_ended()

        # 6. nvidia-smi — only when instance alive and enough time has passed
        if time.time() - self._last_status_t >= self._STATUS_MIN_SECS:
            self._refresh_status_cache(ssh)
            self._last_status_t = time.time()

    def _fetch_log_lines(self, ssh: InstanceSSH, remote_log: str,
                         final: bool = False) -> None:
        """Fetch log lines since _log_offset via SSH; append to local log file."""
        timeout = 30 if final else 10
        try:
            out = ssh.run(
                f"tail -n +{self._log_offset + 1} {remote_log} 2>/dev/null",
                timeout=timeout)
            if out:
                new_lines = out.splitlines()
                with open(self._local_log, "a") as f:
                    for line in new_lines:
                        f.write(line + "\n")
                self._log_offset += len(new_lines)
        except Exception as e:
            log_debug("fetch_log_lines failed", remote_log=remote_log,
                      offset=self._log_offset, final=final, error=str(e))

    def _get_mtime(self, path: str) -> int:
        out = self._ssh.run(f"stat -c %Y {path} 2>/dev/null", timeout=10)
        return int(out.strip())

    def _remote_file_exists(self, remote_path: str) -> bool:
        out = self._ssh.run(
            f"test -f {remote_path} && echo yes || echo no", timeout=10)
        return out.strip() == "yes"

    def _upload_file(self, local_path: str, remote_path: str) -> None:
        self._ssh.copy_to(local_path, remote_path)

    def _ensure_remote_dir(self, remote_dir: str) -> None:
        self._ssh.run(f"mkdir -p {remote_dir}", timeout=10)

    def _download_checkpoints(self, ssh: InstanceSSH, run_name: str,
                              instance_id: str) -> None:
        """Download checkpoint files (.msgpack + .json) from the remote instance."""
        if not run_name:
            return
        local_output = self._backend.project_root / "output"
        local_output.mkdir(exist_ok=True)
        try:
            listing = ssh.run(
                f"ls /root/project/output/{run_name}*.msgpack "
                f"/root/project/output/{run_name}*.json 2>/dev/null || true",
                timeout=10)
            all_files = [f.strip() for f in listing.splitlines() if f.strip()]
            if not all_files:
                log_debug("no checkpoints found to download", run_name=run_name,
                          instance_id=instance_id)
                return
            msgpacks = [f for f in all_files if f.endswith(".msgpack")]
            stem = self._pick_checkpoint_stem(msgpacks)
            if not stem:
                log_debug("no suitable checkpoint to download", run_name=run_name,
                          instance_id=instance_id)
                return
            files = [f for f in all_files if Path(f).stem == stem]
            for remote_file in files:
                local_file = str(local_output / Path(remote_file).name)
                log_debug("downloading checkpoint", remote=remote_file, local=local_file)
                ssh.copy_from(remote_file, local_file)
            log_debug("checkpoint download complete", run_name=run_name,
                      count=len(files), instance_id=instance_id)
        except Exception as e:
            log_error("checkpoint download failed", exc=e, run_name=run_name,
                      instance_id=instance_id)

    def _refresh_status_cache(self, ssh: InstanceSSH) -> None:
        """Fetch nvidia-smi via SSH and update the cached status."""
        try:
            out = ssh.run(
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
        except Exception as e:
            log_debug("nvidia-smi SSH fetch failed", instance_id=self._instance_id,
                      error=str(e))

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
                instance_id = self._instance_id
                ssh         = self._ssh

            if not instance_id:
                raise RuntimeError("No instance attached to this slot — attach one first.")
            if not ssh:
                raise RuntimeError("No SSH connection to instance.")

            # 0. Reject if a training job for this run_name is already tracked
            with self._lock:
                already_tracked = (self._training_status in ("submitted", "running")
                                   and self._run_name == run_name)
            if already_tracked:
                raise RuntimeError(
                    f"A training job for '{run_name}' is already running on this instance. "
                    "Kill it first."
                )
            try:
                existing = ssh.run(
                    f"pgrep -fa 'train.py' | grep -E -- '--run-name {run_name}( |$)' || true",
                    timeout=10)
                if existing.strip():
                    pid = existing.strip().split()[0]
                    raise RuntimeError(
                        f"Instance already has a running '{run_name}' process (PID {pid}). "
                        "Kill it first."
                    )
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"Could not check for existing processes: {e}")

            # 1. Stage server state before launch so a timeout leaves clean state
            Path(self._local_log).unlink(missing_ok=True)
            with self._lock:
                self._remote_log      = remote_log
                self._run_name        = run_name
                self._log_file        = self._local_log
                self._training_status = "submitted"
                self._submitted_at    = time.time()
                self._absent_training = 0
                self._log_offset      = 0
                self._log_complete    = False

            # 2. Copy training files to /root/project/ via SCP
            ssh.run("mkdir -p /root/project/logs", timeout=10)
            project_root = self._backend.project_root
            for fname in ("train.py", "policy.py", "default_params.json"):
                src = project_root / fname
                if src.exists():
                    ssh.copy_to(str(src), f"/root/project/{fname}")

            # 3. Ensure checkpoint is available on the instance
            if chk:
                chk = self._ensure_remote_checkpoint(chk)

            # 4. Build training command and launch via SSH (background + disown)
            python_prefix = " ".join(self.python_cmd)
            cmd = f"cd /root/project && {python_prefix} {script} --run-name {run_name}"
            for key, val in params.items():
                cmd += f" --{key.replace('_', '-')} {val}"
            if chk:
                cmd += f" --checkpoint {chk}"
            cmd += f" > {remote_log} 2>&1 </dev/null & disown"
            # Launch via SSH in background thread — non-blocking
            _ssh_execute_async(ssh, cmd, instance_id=instance_id)
            # Poll thread will confirm the process started and transition to "running"
            conn.sendall(f"ok {instance_id}\n".encode())
        except Exception as e:
            log_error("_cmd_submit failed", exc=e)
            conn.sendall(f"error {e}\n".encode())
        finally:
            conn.close()

    def _cmd_cancel(self, conn) -> None:
        with self._lock:
            instance_id = self._instance_id
            run_name    = self._run_name
        try:
            if not instance_id:
                conn.sendall(b"error no job running\n")
                return
            # Kill only the training process — leave the instance running
            with self._lock:
                ssh = self._ssh
            if ssh is None:
                conn.sendall(b"error no SSH connection\n")
                return
            _ssh_execute_async(ssh, f"pkill -f 'train.py.*--run-name {run_name}' || true",
                               instance_id=instance_id)
            conn.sendall(b"ok\n")
        except Exception as e:
            log_error("_cmd_cancel failed", exc=e, instance_id=instance_id,
                      run_name=run_name)
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
        # Attach existing instances before accepting dashboard connections
        try:
            self._attach_untracked_instances()
        except Exception as e:
            log_error("_attach_untracked_instances failed (startup)", exc=e)

        self._backend_server = _BackendServer(
            self._servers, sock_path, backend_id,
        )
        self._backend_server.start()
        threading.Thread(target=self._poll_instances_loop, daemon=True,
                         name="vast-instance-poll").start()

    def _poll_instances_loop(self) -> None:
        """Background thread: discover running Vast.ai instances every 30s
        and attach any untracked ones to idle slots."""
        while True:
            time.sleep(30)
            try:
                self._attach_untracked_instances()
            except Exception as e:
                log_error("_attach_untracked_instances failed", exc=e)

    def _attach_untracked_instances(self) -> None:
        """Discover running Vast.ai instances not yet tracked and attach them to
        idle slots. Alive checking and log/status fetching are handled by each
        slot's _poll() thread."""
        instances = _show_instances()
        running = [
            i for i in instances
            if i.get("actual_status") == "running"
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

            # Build SSH connection and discover running training process (if any)
            instance_id_str = str(info["id"])
            ssh = InstanceSSH.from_instance(info)
            run_name = info.get("label") or f"instance-{instance_id_str}"
            training_found = False
            try:
                ps_out = ssh.run(
                    "ps -eo args | grep 'train.py' | grep -v grep | head -1",
                    timeout=10)
                parts = ps_out.split()
                training_found = bool(parts)
                if "--run-name" in parts:
                    run_name = parts[parts.index("--run-name") + 1]
            except Exception as e:
                log_debug("SSH process discovery failed during attach",
                          instance_id=instance_id_str, error=str(e))

            total_cost = _instance_total_cost(instance_id_str)
            with idle._lock:
                idle._instance_id     = instance_id_str
                idle._ssh             = ssh
                idle._cost_per_hour   = float(info.get("dph_total", 0.0))
                idle._total_cost      = total_cost
                idle._run_name        = run_name if training_found else None
                idle._log_file        = idle._local_log if training_found else ""
                idle._remote_log      = "" if training_found else None
                idle._instance_alive  = True
                idle._absent_polls    = 0
                idle._training_status = "running" if training_found else "idle"
                idle._submitted_at    = 0.0
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
