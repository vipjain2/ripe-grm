"""RunPod cloud GPU backend server.

Each "GPU slot" maps to one RunPod pod. Slots are idle until a job is
submitted or an existing pod is discovered. Common server infrastructure
lives in compute_backend_server.py.

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
import subprocess
import sys
import threading
import time
from pathlib import Path

import runpod

from ripe_grm.compute_backend_server import _BackendServer, _BaseGPUServer
from ripe_grm.dashboard_log import log_debug, log_error

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


# ---------------------------------------------------------------------------
# SSH helper — same pattern as vastai_backend.InstanceSSH
# ---------------------------------------------------------------------------

_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=10",
]


class InstanceSSH:
    """SSH connection to a RunPod pod."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def run(self, cmd: str, timeout: int = 30) -> str:
        result = subprocess.run(
            ["ssh", f"-p{self.port}", *_SSH_OPTS, f"root@{self.host}", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        return result.stdout.strip()

    def copy_to(self, local_path: str, remote_path: str, timeout: int = 60) -> None:
        result = subprocess.run(
            ["scp", f"-P{self.port}", *_SSH_OPTS, local_path,
             f"root@{self.host}:{remote_path}"],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

    def copy_from(self, remote_path: str, local_path: str, timeout: int = 120) -> None:
        result = subprocess.run(
            ["scp", f"-P{self.port}", *_SSH_OPTS,
             f"root@{self.host}:{remote_path}", local_path],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

    def wait_ready(self, timeout: int = 120) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self.run("echo ok", timeout=10)
                return
            except Exception:
                time.sleep(5)
        raise TimeoutError(f"SSH {self.host}:{self.port} not ready within {timeout}s")

    @classmethod
    def from_pod(cls, pod: dict) -> "InstanceSSH":
        """Build from a RunPod pod info dict.

        RunPod exposes SSH via runtime.ports — find the entry mapping
        privatePort=22 to a public IP/port.
        """
        runtime = pod.get("runtime") or {}
        for p in runtime.get("ports") or []:
            if p.get("privatePort") == 22 and p.get("isIpPublic"):
                return cls(p["ip"], int(p["publicPort"]))
        # Fallback: top-level publicIp with portMappings (older SDK shape)
        host = pod.get("publicIp")
        if host:
            mappings = pod.get("portMappings") or {}
            return cls(host, int(mappings.get("22", 22)))
        raise RuntimeError(f"Pod {pod.get('id')} has no SSH endpoint (runtime not ready?)")


def _ssh_execute_async(ssh: InstanceSSH, cmd: str, pod_id: str = "") -> None:
    """Fire-and-forget with connection error detection.

    Uses ssh -f which backgrounds only after successful authentication.
    Connection/auth failures exit immediately with a non-zero code.
    """
    def _run() -> None:
        try:
            result = subprocess.run(
                ["ssh", "-f", f"-p{ssh.port}", *_SSH_OPTS, f"root@{ssh.host}", cmd],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, text=True, timeout=30,
            )
            if result.returncode != 0 and result.stderr.strip():
                log_error("ssh execute failed", pod_id=pod_id,
                          cmd=cmd[:80], error=result.stderr.strip()[:500])
        except subprocess.TimeoutExpired:
            log_error("ssh execute timed out", pod_id=pod_id, cmd=cmd[:80])
    threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# _RunPodGPUServer
# ---------------------------------------------------------------------------

class _RunPodGPUServer(_BaseGPUServer):
    """One virtual GPU slot backed by a RunPod pod.

    Idle when no pod is attached; active while a pod is running.
    The background poll thread handles all SSH work: alive checks,
    log fetching, and nvidia-smi updates.
    """

    _POLL_INTERVAL   = 10
    _STATUS_MIN_SECS = 30

    @property
    def python_cmd(self) -> list[str]:
        return ["uv", "run", "--project", "/app", "python", "-u"]

    def __init__(self, slot_id: int, backend: "RunPodBackend"):
        super().__init__(slot_id, f"RunPod Slot {slot_id}",
                         f"/tmp/drl_runpod_{slot_id}.sock")
        self._backend: "RunPodBackend" = backend
        self._ssh: InstanceSSH | None = None
        self._remote_log: str | None = None
        self._training_status: str = "idle"
        self._submitted_at: float = 0.0
        self._absent_training: int = 0
        self._ssh_fail_polls: int = 0
        self._local_log: str = f"/tmp/drl_runpod_{slot_id}.log"
        self._log_offset: int = 0
        self._log_complete: bool = False
        self._status_cache: dict = {"util_pct": 0, "temp_c": 0,
                                     "mem_used_mb": 0, "mem_total_mb": 0}
        self._status_lock = threading.Lock()
        self._last_status_t: float = 0.0
        self._status_fail_count: int = 0

    # -- Job-state helpers (called with self._lock held) ---------------------

    def _job_alive(self) -> bool:
        return (self._instance_state == self.INSTANCE_RUNNING
                and self._training_status in ("submitted", "running", "downloading"))

    def _job_id(self) -> str:
        return str(self._instance_id) if self._instance_id else "-"

    def _on_job_ended(self) -> None:
        self._training_status = "idle"
        self._submitted_at = 0.0
        self._absent_training = 0
        self._ssh_fail_polls = 0
        self._log_offset = 0
        self._log_complete = True
        self._remote_log = None

    def _on_instance_detached(self) -> None:
        super()._on_instance_detached()
        self._ssh = None
        with self._status_lock:
            self._status_cache = {"util_pct": 0, "temp_c": 0,
                                  "mem_used_mb": 0, "mem_total_mb": 0}
            self._status_fail_count = 0

    # -- Session restore -----------------------------------------------------

    def _build_ssh_from_info(self, info: dict) -> InstanceSSH:
        return InstanceSSH.from_pod(info)

    def _apply_restored_state(self, instance_id: str, ssh: InstanceSSH,
                              run_name: str, info: dict, entry: dict,
                              training_alive: bool) -> None:
        """Hook called by _BaseGPUServer._restore_from_state after the pgrep
        probe. Writes restored slot state under self._lock. If the training
        process is no longer alive on the pod, drop straight to "downloading"
        so the next poll tick runs the final log + checkpoint download path.
        """
        Path(self._local_log).unlink(missing_ok=True)
        with self._lock:
            self._instance_id     = instance_id
            self._instance_state  = self.INSTANCE_RUNNING
            self._absent_polls    = 0
            self._ssh             = ssh
            self._run_name        = run_name
            self._log_file        = self._local_log
            self._remote_log      = ""   # rediscovered by _poll step 3
            self._cost_per_hour   = float(info.get("costPerHr", 0.0))
            self._submitted_at    = 0.0
            self._absent_training = 0
            self._ssh_fail_polls  = 0
            self._log_offset      = 0
            self._log_complete    = False
            if training_alive:
                self._training_status = "running"
            else:
                self._training_status = "downloading"

    # register_job is intentionally not overridden. See the note in
    # vastai_backend._VastGPUServer — register_existing is only exercised by
    # dashboard._attach_running_processes (local nvidia-smi scan), never for
    # remote pods. Dashboard restart uses the reattach command.

    # -- Heartbeat overrides -------------------------------------------------

    def _check_instance_alive(self, instance_id: str) -> dict | None:
        pod = _get_pod(instance_id)
        if pod and pod.get("desiredStatus") == "RUNNING":
            return pod
        return None

    def _on_heartbeat_alive(self, info: dict) -> None:
        self._cost_per_hour = float(info.get("costPerHr", self._cost_per_hour))

    def _discover_extra(self) -> dict:
        return {}

    # -- Background poll -----------------------------------------------------

    def _poll(self) -> None:
        with self._lock:
            instance_id = self._instance_id
            ssh = self._ssh
            remote_log = self._remote_log
            run_name = self._run_name or ""
            training_status = self._training_status

        if not instance_id:
            return

        # 0. Rediscovery: attached slot that thinks it's idle may have a
        # training process that started out-of-band (or was lost to an
        # earlier buggy cleanup). Cheap SSH scan; only runs when idle.
        if training_status == "idle" and ssh is not None:
            self._try_rediscover_training(ssh, instance_id)
            with self._lock:
                training_status = self._training_status
                run_name = self._run_name or ""
                remote_log = self._remote_log

        # 1. Instance heartbeat
        is_alive = self._heartbeat()

        if not is_alive:
            if self._is_instance_gone():
                if not self._log_complete:
                    try:
                        self._fetch_log_lines(ssh, remote_log, final=True)
                        self._download_checkpoints(ssh, run_name, instance_id)
                    except Exception:
                        pass
                    self._log_complete = True
                with self._lock:
                    if self._instance_id == instance_id:
                        self._on_instance_detached()
            return

        # 2. Training process state machine
        is_training_alive = False
        if training_status in ("submitted", "running"):
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
                # SSH outage: do NOT conclude the process ended. Keep the
                # tracked state and retry next poll. Surface a single error
                # at the 5-fail mark so the user notices, then stay quiet.
                if ssh_fails == 5:
                    log_error("SSH unreachable for 5 polls; retaining state",
                              pod_id=instance_id, run_name=run_name, error=str(e))
                else:
                    log_debug("pgrep SSH check failed; retaining state",
                              pod_id=instance_id, run_name=run_name,
                              fail_count=ssh_fails, error=str(e))
                process_found = None

            # Broad scan also acts as a confirmation signal: if the specific
            # pgrep returned empty but train.py is still running on the pod,
            # the specific check almost certainly had a transient miss (e.g.
            # a brief uv→python cmdline blip during JAX compile). Reclassify
            # as "unknown" (None) so we retain state instead of marching the
            # absent counter toward "downloading". Gated on "running" — in
            # "submitted" state the broad scan could pick up a stale process
            # from a previous run, overwriting the newly submitted name.
            detected_name: str | None = None
            if process_found is False and training_status == "running":
                try:
                    broad_out = ssh.run(
                        "ps -eo args | grep 'train.py' | grep -v grep | head -1",
                        timeout=10)
                    if broad_out.strip():
                        parts = broad_out.strip().split()
                        if "--run-name" in parts:
                            seen = parts[parts.index("--run-name") + 1]
                            if seen == run_name:
                                # Same run still alive — specific pgrep blipped
                                process_found = None
                            else:
                                detected_name = seen
                        else:
                            # train.py present but couldn't parse run-name —
                            # retain state, better to wait one more poll than
                            # fast-track a healthy process to "downloading".
                            process_found = None
                except Exception as e:
                    log_debug("Broad process scan failed", pod_id=instance_id,
                              error=str(e))

            with self._lock:
                if self._instance_id == instance_id:
                    if process_found:
                        self._training_status = "running"
                        self._absent_training = 0
                    elif detected_name:
                        log_debug("Switching tracked run to detected process",
                                  was=run_name, now=detected_name, pod_id=instance_id)
                        Path(self._local_log).unlink(missing_ok=True)
                        self._run_name = detected_name
                        self._log_file = self._local_log
                        self._remote_log = ""
                        self._training_status = "running"
                        self._absent_training = 0
                        self._log_offset = 0
                        self._log_complete = False
                    elif process_found is False:
                        if training_status == "submitted":
                            if time.time() - self._submitted_at > 60.0:
                                log_error("Training process never confirmed within 60s",
                                          pod_id=instance_id, run_name=run_name)
                                self._training_status = "idle"
                                self._run_name = None
                                self._log_file = ""
                        elif training_status == "running":
                            # Threshold of 5 (~25s with default poll interval)
                            # gives JAX JIT compile and other CPU-pinning
                            # startup work room to breathe before we conclude
                            # the run actually ended. Combined with the broad
                            # scan reclassifying transient pgrep misses as
                            # "unknown", this kills the class of bugs where a
                            # healthy training process gets prematurely flipped
                            # to "downloading".
                            self._absent_training += 1
                            if self._absent_training >= 5:
                                self._training_status = "downloading"
                    is_training_alive = self._training_status in ("submitted", "running")

        # 3. Discover remote log path
        if not remote_log and run_name:
            try:
                found = ssh.run(
                    f"ls -t /workspace/project/logs/{run_name}*.log 2>/dev/null | head -1",
                    timeout=10).strip()
                remote_log = found or "/var/log/syslog"
                with self._lock:
                    if self._instance_id == instance_id:
                        self._remote_log = remote_log
            except Exception as e:
                log_debug("remote log discovery failed", pod_id=instance_id,
                          run_name=run_name, error=str(e))

        if not remote_log:
            return

        # 4. Fetch log lines while training alive
        if is_training_alive:
            self._fetch_log_lines(ssh, remote_log)

        # 5. Download checkpoints + final log when training ended
        is_downloading = training_status == "downloading" or (
            training_status in ("submitted", "running") and not is_training_alive
        )
        if is_downloading and not self._log_complete:
            self._append_local_log("[runpod] Downloading final logs from pod...")
            self._fetch_log_lines(ssh, remote_log, final=True)
            self._append_local_log("[runpod] Downloading checkpoints from pod...")
            self._download_checkpoints(ssh, run_name, instance_id)
            self._append_local_log("[runpod] Download complete.")
            with self._lock:
                if self._instance_id == instance_id:
                    self._on_job_ended()

        # 6. nvidia-smi
        if time.time() - self._last_status_t >= self._STATUS_MIN_SECS:
            self._refresh_status_cache(ssh)
            self._last_status_t = time.time()

    def _append_local_log(self, message: str) -> None:
        """Append a status message to the local log file."""
        with open(self._local_log, "a") as f:
            f.write(message + "\n")

    def _fetch_log_lines(self, ssh: InstanceSSH, remote_log: str,
                         final: bool = False) -> None:
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
                              pod_id: str) -> None:
        if not run_name:
            return
        local_output = self._backend.project_root / "output"
        local_output.mkdir(exist_ok=True)
        try:
            listing = ssh.run(
                f"ls /workspace/project/output/{run_name}*.msgpack "
                f"/workspace/project/output/{run_name}*.json 2>/dev/null || true",
                timeout=10)
            all_files = [f.strip() for f in listing.splitlines() if f.strip()]
            if not all_files:
                log_debug("no checkpoints found to download", run_name=run_name,
                          pod_id=pod_id)
                return
            msgpacks = [f for f in all_files if f.endswith(".msgpack")]
            stem = self._pick_checkpoint_stem(msgpacks)
            if not stem:
                log_debug("no suitable checkpoint to download", run_name=run_name,
                          pod_id=pod_id)
                return
            files = [f for f in all_files if Path(f).stem == stem]
            for remote_file in files:
                local_file = str(local_output / Path(remote_file).name)
                log_debug("downloading checkpoint", remote=remote_file, local=local_file)
                ssh.copy_from(remote_file, local_file)
            log_debug("checkpoint download complete", run_name=run_name,
                      count=len(files), pod_id=pod_id)
        except Exception as e:
            log_error("checkpoint download failed", exc=e, run_name=run_name,
                      pod_id=pod_id)

    def _refresh_status_cache(self, ssh: InstanceSSH) -> None:
        try:
            out = ssh.run(
                "nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,"
                "memory.used,memory.total --format=csv,noheader,nounits",
                timeout=12,
            )
            util, temp, mem_used, mem_total = [x.strip() for x in out.split(",")]
            with self._status_lock:
                self._status_cache = {
                    "util_pct":     int(util),
                    "temp_c":       int(temp),
                    "mem_used_mb":  int(mem_used),
                    "mem_total_mb": int(mem_total),
                }
                self._status_fail_count = 0
        except Exception as e:
            with self._status_lock:
                self._status_fail_count += 1
                # After 3 consecutive failures (~90s) the cached values are
                # stale enough to mislead. Zero them so the GPU panel shows
                # the SSH outage instead of frozen numbers.
                if self._status_fail_count >= 3:
                    self._status_cache = {"util_pct": 0, "temp_c": 0,
                                          "mem_used_mb": 0, "mem_total_mb": 0}
            log_debug("nvidia-smi SSH fetch failed", pod_id=self._instance_id,
                      error=str(e))

    def _try_rediscover_training(self, ssh: InstanceSSH,
                                  instance_id: str) -> None:
        """Scan the pod for an out-of-band train.py --run-name process.

        Called from _poll when the slot is attached but marked idle. If a
        matching process is found, the slot is promoted back to "running"
        with _run_name extracted from the process args. Silent on SSH
        errors — we'll retry next poll.
        """
        try:
            ps_out = ssh.run(
                "ps -eo args | grep 'train.py' | grep -- '--run-name' "
                "| grep -v grep | head -1",
                timeout=10)
        except Exception as e:
            log_debug("rediscovery ps scan failed", pod_id=instance_id, error=str(e))
            return

        parts = ps_out.split()
        if "--run-name" not in parts:
            return
        idx = parts.index("--run-name")
        if idx + 1 >= len(parts):
            return
        discovered = parts[idx + 1]

        with self._lock:
            # Ensure we still own this slot and it's still idle before promoting
            if self._instance_id != instance_id or self._training_status != "idle":
                return
            Path(self._local_log).unlink(missing_ok=True)
            self._run_name        = discovered
            self._log_file        = self._local_log
            self._remote_log      = ""   # discovered on next poll
            self._training_status = "running"
            self._submitted_at    = 0.0
            self._absent_training = 0
            self._ssh_fail_polls  = 0
            self._log_offset      = 0
            self._log_complete    = False
        log_debug("rediscovered train.py process",
                  pod_id=instance_id, run_name=discovered)

    # -- Hardware-specific commands -----------------------------------------

    def _gpu_stats(self) -> dict:
        with self._status_lock:
            return dict(self._status_cache)

    def _cmd_submit(self, conn, json_str: str) -> None:
        try:
            cfg = json.loads(json_str)
            run_name = cfg["run_name"]
            script = cfg["script"]
            params = cfg.get("params", {})
            chk = cfg.get("checkpoint")
            log_name = Path(cfg.get("log_file", f"{run_name}.log")).name
            remote_log = f"/workspace/project/logs/{log_name}"

            with self._lock:
                instance_id = self._instance_id
                ssh = self._ssh

            if not instance_id:
                raise RuntimeError("No pod attached to this slot — attach one first.")
            if not ssh:
                raise RuntimeError("No SSH connection to pod.")

            with self._lock:
                already_tracked = (self._training_status in ("submitted", "running")
                                   and self._run_name == run_name)
            if already_tracked:
                raise RuntimeError(
                    f"A training job for '{run_name}' is already running on this pod. "
                    "Kill it first."
                )
            try:
                existing = ssh.run(
                    f"pgrep -fa 'train.py' | grep -E -- '--run-name {run_name}( |$)' || true",
                    timeout=10)
                if existing.strip():
                    pid = existing.strip().split()[0]
                    raise RuntimeError(
                        f"Pod already has a running '{run_name}' process (PID {pid}). "
                        "Kill it first."
                    )
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"Could not check for existing processes: {e}")

            Path(self._local_log).unlink(missing_ok=True)
            with self._lock:
                self._remote_log = remote_log
                self._run_name = run_name
                self._log_file = self._local_log
                self._training_status = "submitted"
                self._submitted_at = time.time()
                self._absent_training = 0
                self._log_offset = 0
                self._log_complete = False

            ssh.run("mkdir -p /workspace/project/logs", timeout=10)
            project_root = self._backend.project_root
            for fname in ("train.py", "policy.py", "default_params.json"):
                src = project_root / fname
                if src.exists():
                    ssh.copy_to(str(src), f"/workspace/project/{fname}")

            if chk:
                chk = self._ensure_remote_checkpoint(
                    chk, remote_dir="/workspace/project/output")

            python_prefix = " ".join(self.python_cmd)
            cmd = f"cd /workspace/project && {python_prefix} {script} --run-name {run_name}"
            for key, val in params.items():
                cmd += f" --{key.replace('_', '-')} {val}"
            if chk:
                cmd += f" --checkpoint {chk}"
            cmd += f" > {remote_log} 2>&1 </dev/null & disown"
            _ssh_execute_async(ssh, cmd, pod_id=instance_id)
            conn.sendall(f"ok {instance_id}\n".encode())
        except Exception as e:
            log_error("_cmd_submit failed", exc=e)
            conn.sendall(f"error {e}\n".encode())
        finally:
            conn.close()

    def _cmd_cancel(self, conn) -> None:
        with self._lock:
            instance_id = self._instance_id
            run_name = self._run_name
        try:
            if not instance_id:
                conn.sendall(b"error no job running\n")
                return
            with self._lock:
                ssh = self._ssh
            if ssh is None:
                conn.sendall(b"error no SSH connection\n")
                return
            _ssh_execute_async(ssh, f"pkill -f 'train.py.*--run-name {run_name}' || true",
                               pod_id=instance_id)
            conn.sendall(b"ok\n")
        except Exception as e:
            log_error("_cmd_cancel failed", exc=e, pod_id=instance_id,
                      run_name=run_name)
            conn.sendall(f"error {e}\n".encode())
        finally:
            conn.close()

    def _cmd_logs(self, conn) -> None:
        local_log = self._local_log
        try:
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
# RunPodBackend
# ---------------------------------------------------------------------------

class RunPodBackend:
    """Manages a pool of virtual GPU slots backed by RunPod pods.
    Pure server — use BackendClient from compute_backend_client.py to interact."""

    def __init__(self,
                 num_slots:    int        = 1,
                 project_root: Path | None = None,
                 sock_path:    str        = DEFAULT_RUNPOD_SOCK,
                 backend_id:   str        = "runpod"):
        _init_api_key()
        self.project_root = project_root or Path.cwd()
        self._servers: dict[int, _RunPodGPUServer] = {
            i: _RunPodGPUServer(i, self) for i in range(num_slots)
        }
        for server in self._servers.values():
            server.start()
        try:
            self._attach_untracked_pods()
        except Exception as e:
            log_error("_attach_untracked_pods failed (startup)", exc=e)

        self._backend_server = _BackendServer(
            self._servers, sock_path, backend_id,
        )
        self._backend_server.start()
        threading.Thread(target=self._poll_pods_loop, daemon=True,
                         name="runpod-pod-poll").start()

    def _poll_pods_loop(self) -> None:
        while True:
            time.sleep(30)
            try:
                self._attach_untracked_pods()
            except Exception as e:
                log_error("_attach_untracked_pods failed", exc=e)

    def _attach_untracked_pods(self) -> None:
        """Discover running RunPod pods not yet tracked and attach to idle slots."""
        pods = _get_pods()
        # desiredStatus is the target — runtime is populated only once the pod
        # is actually started and SSH is reachable. Skip provisioning pods.
        running = [p for p in pods
                   if p.get("desiredStatus") == "RUNNING" and p.get("runtime")]

        tracked = set()
        for server in self._servers.values():
            with server._lock:
                if server._instance_id:
                    tracked.add(str(server._instance_id))

        for pod in running:
            pod_id = str(pod["id"])
            if pod_id in tracked:
                continue

            idle = next(
                (s for s in self._servers.values() if s._instance_id is None),
                None,
            )
            if idle is None:
                slot_id = max(self._servers) + 1
                idle = _RunPodGPUServer(slot_id, self)
                idle.start()
                self._servers[slot_id] = idle

            try:
                ssh = InstanceSSH.from_pod(pod)
            except Exception as e:
                log_debug("Could not build SSH for pod", pod_id=pod_id, error=str(e))
                continue

            # Scan for an existing train.py --run-name process. Only treat
            # the slot as "running a job" if we can extract a real run_name
            # from the process args — never fall back to the pod's auto-name,
            # which would poison checkpoint paths and tracking.
            discovered_run_name: str | None = None
            try:
                ps_out = ssh.run(
                    "ps -eo args | grep 'train.py' | grep -- '--run-name' "
                    "| grep -v grep | head -1",
                    timeout=10)
                parts = ps_out.split()
                if "--run-name" in parts:
                    idx = parts.index("--run-name")
                    if idx + 1 < len(parts):
                        discovered_run_name = parts[idx + 1]
            except Exception as e:
                log_debug("SSH process discovery failed during attach",
                          pod_id=pod_id, error=str(e))

            training_found = discovered_run_name is not None

            with idle._lock:
                idle._instance_id = pod_id
                idle._instance_state = idle.INSTANCE_RUNNING
                idle._absent_polls = 0
                idle._ssh = ssh
                idle._cost_per_hour = float(pod.get("costPerHr", 0.0))
                idle._run_name = discovered_run_name
                idle._log_file = idle._local_log if training_found else ""
                idle._remote_log = "" if training_found else None
                idle._training_status = "running" if training_found else "idle"
                idle._submitted_at = 0.0
                idle._absent_training = 0
                idle._log_offset = 0
                idle._log_complete = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _sock_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUNPOD_SOCK
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
