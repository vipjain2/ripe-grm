"""Common server-side infrastructure shared across compute backends.

_BaseGPUServer       — minimal socket + protocol skeleton. Subclasses
                       implement hardware-specific commands (_cmd_submit,
                       _cmd_cancel, _cmd_logs) and the job-state helpers
                       (_job_alive, _job_id). Used directly by the local
                       NVIDIA backend. Lock invariant: _job_alive(),
                       _job_id(), and _make_handle() must always be called
                       with self._lock held.

InstanceSSH          — plain SSH connection wrapper used by cloud backends.

_BaseCloudGPUServer  — layered on top of _BaseGPUServer: adds instance
                       state, heartbeat, background poll, training-process
                       state machine, log fetching, checkpoint download,
                       and session restore. Cloud providers subclass this
                       and implement four small hooks.

_BaseCloudBackend    — cloud backend lifecycle: slot pool, startup
                       discovery, and instance poll loop.

_BackendServer       — backend-level socket server for discovery,
                       running-job queries, and the dashboard-driven
                       `reattach` command. Works with any dict[int,
                       _BaseGPUServer].

State persistence lives on the dashboard side (runs_state.json). Backends
are stateless on disk; they rebuild slot state on each restart via
_attach_untracked_instances (discovery) and _cmd_reattach (dashboard replay).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

from ripe_grm.compute_backend_client import SocketJobHandle
from ripe_grm.dashboard_log import log_debug, log_error


# ---------------------------------------------------------------------------
# Cloud SSH infrastructure
# ---------------------------------------------------------------------------

_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=10",
]


class InstanceSSH:
    """SSH connection to a cloud instance.

    Used for all remote commands: monitoring (pgrep, tail, nvidia-smi, ps),
    launch and cancel (_ssh_execute_async), and file transfer (copy_to/SCP).
    Provider-specific factories live in each cloud backend module.
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


def _ssh_execute_async(ssh: "InstanceSSH", cmd: str, instance_id: str = "") -> None:
    """Fire-and-forget SSH execution with connection error detection.

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
                log_error("ssh execute failed", instance_id=instance_id,
                          cmd=cmd[:80], error=result.stderr.strip()[:500])
        except subprocess.TimeoutExpired:
            log_error("ssh execute timed out", instance_id=instance_id, cmd=cmd[:80])
    threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# SSH process-discovery helpers
# ---------------------------------------------------------------------------
#
# Two independent probes are used throughout the cloud backend:
#
#   _ssh_find_run       — specific: "is a train.py --run-name <name> process
#                         alive right now?" Used during _poll, submit, and
#                         restore. Raises on SSH error so callers can apply
#                         their own fail-tolerance policy.
#
#   _ssh_scan_train_run_name — broad: "is *any* train.py --run-name process
#                         alive on this instance, and if so, what run-name?"
#                         Used during startup discovery and idle rediscovery.
#                         Swallows SSH errors and returns None (debug-logged).

def _ssh_find_run(ssh: "InstanceSSH", run_name: str, timeout: int = 10) -> str:
    """Return the raw pgrep output (format: 'PID cmdline...') for the
    specific train.py --run-name <run_name> process on the instance, or
    empty string if no match. Raises on SSH/network error."""
    return ssh.run(
        f"pgrep -fa 'train.py' | grep -E -- '--run-name {run_name}( |$)' || true",
        timeout=timeout,
    )


def _ssh_scan_train_run_name(ssh: "InstanceSSH",
                             instance_id: str = "") -> str | None:
    """Scan the instance for any running train.py process that carries
    a --run-name argument, and return that run_name. Returns None if no
    such process is found, or if the SSH call fails (debug-logged).
    """
    try:
        out = ssh.run(
            "ps -eo args | grep 'train.py' | grep -- '--run-name' "
            "| grep -v grep | head -1",
            timeout=10,
        )
    except Exception as e:
        log_debug("SSH train.py scan failed",
                  instance_id=instance_id, error=str(e))
        return None
    parts = out.split()
    if "--run-name" not in parts:
        return None
    idx = parts.index("--run-name")
    if idx + 1 >= len(parts):
        return None
    return parts[idx + 1]


# ---------------------------------------------------------------------------
# Socket + protocol helpers
# ---------------------------------------------------------------------------

def _read_line(conn: socket.socket) -> str:
    """Read one newline-terminated line from a Unix socket and return the
    decoded payload (without the trailing newline). Returns empty string on
    EOF before a newline arrives."""
    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(256)
        if not chunk:
            return ""
        buf += chunk
    return buf.split(b"\n", 1)[0].decode().strip()


def _handle_to_dict(h: SocketJobHandle, sock_path: str, backend_id: str) -> dict:
    """Serialize a SocketJobHandle to the JSON dict shape expected by the
    dashboard's BackendClient. Used by every _BackendServer command that
    returns a running-job handle. Callers pass the owning slot's sock_path
    explicitly — it isn't part of the handle's public interface."""
    return {
        "run_name":   h.run_name,
        "id":         h.id,
        "gpu_id":     h.gpu_id,
        "log_file":   h.log_file,
        "sock_path":  sock_path,
        "backend_id": backend_id,
    }


# ---------------------------------------------------------------------------
# _BaseGPUServer
# ---------------------------------------------------------------------------

class _BaseGPUServer(ABC):
    """Per-GPU socket server skeleton.

    Minimal generic base: accept socket connections, dispatch commands,
    and delegate the hardware-specific work (submit / cancel / logs /
    job-state) to subclasses. Used directly by the local NVIDIA backend.
    Cloud providers subclass _BaseCloudGPUServer instead, which layers
    SSH/poll/restore infrastructure on top.
    """

    def __init__(self, index: int, name: str, sock_path: str):
        self._index    = index
        self.name      = name
        self.sock_path = sock_path
        self._lock     = threading.Lock()
        self._run_name: str | None = None
        self._log_file: str = ""
        self._ready    = threading.Event()

    @property
    def python_cmd(self) -> list[str]:
        """Command prefix to invoke Python for a training job.
        Returns [sys.executable, "-u"] by default; cloud backends override this."""
        return [sys.executable, "-u"]

    def start(self) -> None:
        t = threading.Thread(target=self._serve, daemon=True,
                             name=f"gpu-server-{self._index}")
        t.start()
        self._ready.wait(timeout=5.0)

    # -- Abstract / overridable interface ------------------------------------

    def register_job(self, run_name: str, job_id: str,
                     log_file: str) -> "SocketJobHandle | None":
        """Re-attach to an already-running job by opaque job id.

        Used by the `register_existing` socket command, which is driven by
        dashboard._attach_running_processes — a scan of local nvidia-smi
        compute-apps. Only the local backend overrides this; cloud backends
        never see a local PID for a remote instance, so the default no-op
        applies. Dashboard-restart reattach for cloud slots goes through
        `_cmd_reattach` / `_restore_from_state` instead.
        """
        return None

    def _restore_from_state(self, entry: dict) -> "SocketJobHandle | None":
        """Rebuild slot state from a persisted runs_state.json entry.
        Default no-op — cloud backends override. Called by the dashboard-
        driven reattach path after a dashboard restart."""
        return None

    @abstractmethod
    def _job_alive(self) -> bool:
        """True if the current job is still running.
        MUST be called with self._lock held."""
        ...

    @abstractmethod
    def _job_id(self) -> str:
        """Opaque job ID string (e.g. PID, cloud instance ID).
        MUST be called with self._lock held."""
        ...

    def _on_job_ended(self) -> None:
        """Called (with self._lock held) when a job is found to have ended.
        Override to clear subclass-specific job state (e.g. _pid, _process)."""
        pass

    def _gpu_stats(self) -> dict:
        """Return GPU utilisation stats (util_pct, temp_c, mem_used_mb, mem_total_mb).
        Override in backends that can report hardware stats."""
        return {"util_pct": 0, "temp_c": 0, "mem_used_mb": 0, "mem_total_mb": 0}

    def _cmd_status(self, conn: socket.socket) -> None:
        """Return GPU status over the socket. Cloud backends override to
        add instance fields (instance_id, cost_per_hour, etc.)."""
        try:
            conn.sendall((json.dumps(self._gpu_stats()) + "\n").encode())
        except Exception as e:
            conn.sendall((json.dumps({"error": str(e)}) + "\n").encode())
        finally:
            conn.close()

    @abstractmethod
    def _cmd_submit(self, conn: socket.socket, json_str: str) -> None: ...

    @abstractmethod
    def _cmd_cancel(self, conn: socket.socket) -> None: ...

    @abstractmethod
    def _cmd_logs(self, conn: socket.socket) -> None: ...

    # -- Common implementation -----------------------------------------------

    def current_handle(self) -> "SocketJobHandle | None":
        with self._lock:
            if not self._job_alive():
                self._run_name = None
                self._log_file = ""
                self._on_job_ended()
                return None
            return self._make_handle()

    def _make_handle(self) -> SocketJobHandle:
        """Build a SocketJobHandle from current state.
        MUST be called with self._lock held."""
        return SocketJobHandle(
            run_name  = self._run_name or "",
            gpu_id    = self._index,
            log_file  = self._log_file,
            id        = self._job_id(),
            sock_path = self.sock_path,
        )

    def _serve(self) -> None:
        try:
            os.unlink(self.sock_path)
        except FileNotFoundError:
            pass
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.sock_path)
        server.listen(16)
        self._ready.set()
        try:
            while True:
                conn, _ = server.accept()
                threading.Thread(target=self._handle_client, args=(conn,),
                                 daemon=True).start()
        finally:
            server.close()
            try:
                os.unlink(self.sock_path)
            except FileNotFoundError:
                pass

    def _handle_client(self, conn: socket.socket) -> None:
        try:
            cmd = _read_line(conn)
            if cmd == "info":
                self._cmd_info(conn)
            elif cmd == "status":
                self._cmd_status(conn)
            elif cmd == "job_status":
                self._cmd_job_status(conn)
            elif cmd.startswith("submit "):
                self._cmd_submit(conn, cmd[7:])
            elif cmd == "cancel":
                self._cmd_cancel(conn)
            elif cmd == "logs":
                self._cmd_logs(conn)
            else:
                conn.sendall(f"error unknown: {cmd}\n".encode())
                conn.close()
        except Exception as e:
            log_error("_BaseGPUServer._handle_client unhandled exception", exc=e,
                      gpu_index=self._index)
            try:
                conn.close()
            except Exception:
                pass

    def _cmd_info(self, conn: socket.socket) -> None:
        conn.sendall((json.dumps({
            "index":     self._index,
            "name":      self.name,
            "sock_path": self.sock_path,
        }) + "\n").encode())
        conn.close()

    def _cmd_job_status(self, conn: socket.socket) -> None:
        with self._lock:
            running = self._job_alive()
            resp = json.dumps({
                "run_name": self._run_name,
                "id":       self._job_id(),
                "running":  running,
            })
        conn.sendall((resp + "\n").encode())
        conn.close()


# ---------------------------------------------------------------------------
# _BaseCloudGPUServer — shared implementation for SSH-based cloud GPU slots
# ---------------------------------------------------------------------------
#
# Subclasses must override the following class attributes:
#   _SLOT_NAME_PREFIX    — e.g. "Vast Slot"
#   _SOCK_PATH_FMT       — socket path template, contains "{slot_id}"
#   _LOCAL_LOG_FMT       — local log path template, contains "{slot_id}"
#   _REMOTE_PROJECT_DIR  — project dir on the remote instance
#   _REMOTE_FALLBACK_LOG — log path to tail if the run's own log is missing
#   _LOG_PREFIX          — "[vast]" / "[runpod]" for status messages
#
# And the following methods:
#   _check_instance_alive(instance_id) -> info dict | None
#   _build_ssh_from_info(info)         -> InstanceSSH
#   _cost_from_info(info)              -> float   (per-hour rate)
#   _total_cost_from_info(info)        -> float   (optional; default 0)


class _BaseCloudGPUServer(_BaseGPUServer):
    """One virtual GPU slot backed by an SSH-reachable cloud instance.

    Handles the full training lifecycle: heartbeat, submit, poll, log fetch,
    checkpoint download, and session restore. Backend subclasses only provide
    small provider-specific hooks (SDK calls, cost fields, SSH factory).
    """

    # Instance states
    INSTANCE_NONE    = "none"      # no instance attached to this slot
    INSTANCE_RUNNING = "running"   # instance alive and reachable
    INSTANCE_OFFLINE = "offline"   # instance not responding (transient or gone)

    _POLL_INTERVAL:    int = 10
    _STATUS_MIN_SECS:  int = 30
    _ABSENT_THRESHOLD: int = 2     # consecutive failed heartbeats before detaching

    # Class attrs — subclasses override
    _SLOT_NAME_PREFIX:    str = "Cloud Slot"
    _SOCK_PATH_FMT:       str = "/tmp/drl_cloud_{slot_id}.sock"
    _LOCAL_LOG_FMT:       str = "/tmp/drl_cloud_{slot_id}.log"
    _REMOTE_PROJECT_DIR:  str = "/root/project"
    _REMOTE_FALLBACK_LOG: str = "/var/log/syslog"
    _LOG_PREFIX:          str = "[cloud]"

    @property
    def python_cmd(self) -> list[str]:
        return ["uv", "run", "--project", "/app", "python", "-u"]

    def __init__(self, slot_id: int, backend):
        super().__init__(
            slot_id,
            f"{self._SLOT_NAME_PREFIX} {slot_id}",
            self._SOCK_PATH_FMT.format(slot_id=slot_id),
        )
        self._backend = backend
        # Instance state — heartbeat loop maintains this
        self._instance_id:    str | None = None
        self._instance_state: str        = self.INSTANCE_NONE
        self._absent_polls:   int        = 0
        self._cost_per_hour:  float      = 0.0
        self._total_cost:     float      = 0.0
        self._ssh: InstanceSSH | None = None
        self._remote_log: str | None = None
        # Training state machine: "idle" | "submitted" | "running" | "downloading"
        self._training_status: str   = "idle"
        self._submitted_at:    float = 0.0
        self._absent_training: int   = 0   # consecutive missing pgrep hits (in "running")
        self._last_ssh_error_t: float = 0.0  # rate-limit pgrep SSH-failure error logs
        # Local log file — poll thread downloads SSH output here; _cmd_logs reads it
        self._local_log:   str  = self._LOCAL_LOG_FMT.format(slot_id=slot_id)
        self._log_offset:  int  = 0   # lines fetched from remote so far
        self._log_complete: bool = False  # True after final download done
        # GPU status cache — poll thread writes; _cmd_status reads
        self._status_cache: dict = {"util_pct": 0, "temp_c": 0,
                                     "mem_used_mb": 0, "mem_total_mb": 0}
        self._status_lock = threading.Lock()
        self._last_status_t:     float = 0.0
        self._status_fail_count: int   = 0

    # -- Provider-specific hooks (subclasses implement) ----------------------

    def _check_instance_alive(self, instance_id: str) -> dict | None:
        raise NotImplementedError

    def _build_ssh_from_info(self, info: dict) -> InstanceSSH:
        raise NotImplementedError

    def _cost_from_info(self, info: dict) -> float:
        """Per-hour cost of the instance from provider info dict."""
        return 0.0

    def _total_cost_from_info(self, info: dict) -> float:
        """Total accumulated cost, if the provider exposes it. Default 0."""
        return 0.0

    def _on_heartbeat_alive(self, info: dict) -> None:
        rate = self._cost_from_info(info)
        if rate:
            self._cost_per_hour = rate
        total = self._total_cost_from_info(info)
        if total:
            self._total_cost = total

    # -- Job-state helpers (called with self._lock held) ---------------------

    def _job_alive(self) -> bool:
        return (self._instance_state == self.INSTANCE_RUNNING
                and self._training_status in ("submitted", "running", "downloading"))

    def _job_id(self) -> str:
        return str(self._instance_id) if self._instance_id else "-"

    def _on_job_ended(self) -> None:
        """Clear training state. Instance may still be alive."""
        self._training_status = "idle"
        self._submitted_at    = 0.0
        self._absent_training = 0
        self._log_offset      = 0
        self._log_complete    = True
        self._remote_log      = None

    def _on_instance_detached(self) -> None:
        """Clear all instance state — slot becomes fully idle.
        Called with self._lock held."""
        self._on_job_ended()
        self._instance_id    = None
        self._instance_state = self.INSTANCE_NONE
        self._absent_polls   = 0
        self._cost_per_hour  = 0.0
        self._total_cost     = 0.0
        self._ssh = None
        with self._status_lock:
            self._status_cache = {"util_pct": 0, "temp_c": 0,
                                  "mem_used_mb": 0, "mem_total_mb": 0}
            self._status_fail_count = 0

    # -- Start / background poll ---------------------------------------------

    def start(self) -> None:
        super().start()
        threading.Thread(target=self._poll_loop, daemon=True,
                         name=f"gpu-poll-{self._index}").start()

    def _poll_loop(self) -> None:
        """Background poll thread: calls _poll() every _POLL_INTERVAL seconds."""
        try:
            self._poll()
        except Exception as e:
            log_error("_poll failed (startup)", exc=e, gpu_index=self._index)
        while True:
            time.sleep(self._POLL_INTERVAL)
            try:
                self._poll()
            except Exception as e:
                log_error("_poll failed", exc=e, gpu_index=self._index)

    # -- Instance heartbeat --------------------------------------------------

    def _heartbeat(self) -> bool:
        """Run one heartbeat check. Returns True if instance is alive.
        Updates _instance_state and _absent_polls.

        _check_instance_alive should return info dict if alive, None if
        confirmed gone, or raise on transient errors (network/API failures).
        Transient errors do NOT increment _absent_polls."""
        with self._lock:
            instance_id = self._instance_id
        if not instance_id:
            return False

        try:
            info = self._check_instance_alive(instance_id)
        except Exception as e:
            # Transient error — don't count against the instance
            log_debug("heartbeat check failed (transient)",
                      instance_id=instance_id, error=str(e))
            return False

        is_alive = info is not None

        with self._lock:
            if self._instance_id != instance_id:
                return False  # slot was reassigned during check
            if is_alive:
                self._absent_polls   = 0
                self._instance_state = self.INSTANCE_RUNNING
                self._on_heartbeat_alive(info)
            else:
                self._absent_polls += 1
                if self._absent_polls >= self._ABSENT_THRESHOLD:
                    self._instance_state = self.INSTANCE_OFFLINE
        return is_alive

    def _is_instance_gone(self) -> bool:
        """True if the instance has been offline long enough to detach.
        Call after _heartbeat() returns False."""
        return self._instance_state == self.INSTANCE_OFFLINE

    # -- Attach ---------------------------------------------------------------

    def _attach(self, instance_id: str, ssh: InstanceSSH, info: dict,
                run_name: str | None, training_status: str) -> None:
        """Bind a cloud instance to this slot and reset all tracking state.

        training_status semantics:
          "idle"        — no training job is active on the instance
          "running"     — a training process is running
          "downloading" — training process ended, slot needs final download

        Used by the discovery path (_attach_untracked_instances) and the
        dashboard-restart restore path (_restore_from_state). The `idle`
        case comes from discovery only — restore always has a run_name.
        """
        has_job = training_status != "idle"
        # Wipe any leftover local log so re-fetching from offset 0 doesn't
        # duplicate lines. Harmless when has_job is False (slot is idle).
        if has_job:
            Path(self._local_log).unlink(missing_ok=True)
        with self._lock:
            self._instance_id     = instance_id
            self._instance_state  = self.INSTANCE_RUNNING
            self._absent_polls    = 0
            self._ssh             = ssh
            self._cost_per_hour   = self._cost_from_info(info)
            self._total_cost      = self._total_cost_from_info(info)
            self._run_name        = run_name
            self._log_file        = self._local_log if has_job else ""
            self._remote_log      = "" if has_job else None
            self._training_status = training_status
            self._submitted_at    = 0.0
            self._absent_training = 0
            self._log_offset      = 0
            self._log_complete    = False

    # -- Training process probe ---------------------------------------------

    def _probe_training(self, ssh: InstanceSSH, run_name: str,
                        training_status: str,
                        instance_id: str) -> tuple[str, str | None]:
        """Ask the remote whether the tracked training process is alive.

        Returns a (verdict, new_run_name) tuple, where verdict is one of:
          "alive"    — the tracked run is still running
          "switched" — a different train.py run is on the instance;
                       new_run_name holds its name
          "unknown"  — SSH failed, or state couldn't be confirmed;
                       caller should retain current state
          "absent"   — no train.py process at all; caller should apply
                       the submitted-timeout / absent-counter logic

        On SSH failure, error logs are rate-limited to once per 60 seconds
        via self._last_ssh_error_t.
        """
        try:
            out = _ssh_find_run(ssh, run_name)
            process_found: bool | None = bool(out.strip())
        except Exception as e:
            now = time.time()
            if now - self._last_ssh_error_t >= 60.0:
                log_error("SSH unreachable; retaining tracked state",
                          instance_id=instance_id, run_name=run_name, error=str(e))
                self._last_ssh_error_t = now
            else:
                log_debug("pgrep SSH check failed; retaining current state",
                          instance_id=instance_id, run_name=run_name, error=str(e))
            return ("unknown", None)

        if process_found:
            return ("alive", None)

        # Specific pgrep came back empty. In "running" state, do a broad scan
        # for any train.py. This catches:
        #   (a) a brief uv→python cmdline blip during JAX JIT compile — the
        #       specific pgrep misses but train.py is still up; reclassify
        #       as "unknown" so we don't march the absent counter.
        #   (b) a different run-name taking over the instance — switch
        #       tracking to it.
        # Skip in "submitted": the process may not have started yet, and a
        # stale broad match could overwrite the newly submitted name.
        if training_status != "running":
            return ("absent", None)

        try:
            broad_out = ssh.run(
                "ps -eo args | grep 'train.py' | grep -v grep | head -1",
                timeout=10)
        except Exception as e:
            log_debug("Broad process scan failed",
                      instance_id=instance_id, error=str(e))
            return ("absent", None)

        if not broad_out.strip():
            return ("absent", None)
        parts = broad_out.strip().split()
        if "--run-name" not in parts:
            # train.py present but couldn't parse run-name — retain state,
            # better to wait one more poll than fast-track to "downloading".
            return ("unknown", None)
        seen = parts[parts.index("--run-name") + 1]
        if seen == run_name:
            # Same run still alive — specific pgrep blipped
            return ("unknown", None)
        return ("switched", seen)

    def _apply_training_verdict(self, instance_id: str, verdict: str,
                                new_run_name: str | None,
                                prev_status: str) -> bool:
        """Apply a _probe_training verdict under self._lock.

        Returns True if training is still considered alive after this call.
        """
        with self._lock:
            if self._instance_id != instance_id:
                return False
            if verdict == "alive":
                self._training_status = "running"
                self._absent_training = 0
            elif verdict == "switched":
                log_debug("Switching tracked run to detected process",
                          was=self._run_name, now=new_run_name,
                          instance_id=instance_id)
                Path(self._local_log).unlink(missing_ok=True)
                self._run_name        = new_run_name
                self._log_file        = self._local_log
                self._remote_log      = ""   # discovered on next poll
                self._training_status = "running"
                self._absent_training = 0
                self._log_offset      = 0
                self._log_complete    = False
            elif verdict == "absent":
                if prev_status == "submitted":
                    if time.time() - self._submitted_at > 60.0:
                        log_error("Training process never confirmed within 60s",
                                  instance_id=instance_id, run_name=self._run_name)
                        self._training_status = "idle"
                        self._run_name        = None
                        self._log_file        = ""
                elif prev_status == "running":
                    # Threshold of 5 (~25s with default poll interval) gives
                    # JAX JIT compile and other CPU-pinning startup work room
                    # to breathe before we conclude the run actually ended.
                    # Combined with the broad scan reclassifying transient
                    # pgrep misses as "unknown", this kills the class of bugs
                    # where a healthy training process gets prematurely
                    # flipped to "downloading".
                    self._absent_training += 1
                    if self._absent_training >= 5:
                        self._training_status = "downloading"
            # verdict == "unknown": retain current state
            return self._training_status in ("submitted", "running")

    # -- Background poll -----------------------------------------------------

    def _poll(self) -> None:
        """Single poll iteration:
        0. Rediscover out-of-band training process (if slot attached but idle)
        1. Instance heartbeat via provider SDK
        2. Training process state machine via SSH pgrep
        3. Discover remote log path if unknown
        4. Fetch new log lines while training is alive
        5. Final log + checkpoint download when training ends
        6. nvidia-smi (throttled to _STATUS_MIN_SECS)
        """
        with self._lock:
            instance_id     = self._instance_id
            ssh             = self._ssh
            remote_log      = self._remote_log
            run_name        = self._run_name or ""
            training_status = self._training_status

        if not instance_id:
            return

        # 0. Rediscover out-of-band train.py process on an idle attached slot.
        if training_status == "idle" and ssh is not None:
            self._try_rediscover_training(ssh, instance_id)
            with self._lock:
                training_status = self._training_status
                run_name        = self._run_name or ""
                remote_log      = self._remote_log

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

        # 2. Training process state machine — pgrep only when not idle
        is_training_alive = False
        if training_status in ("submitted", "running"):
            verdict, new_name = self._probe_training(
                ssh, run_name, training_status, instance_id)
            is_training_alive = self._apply_training_verdict(
                instance_id, verdict, new_name, prev_status=training_status)

        # 3. Discover remote log path when not yet known
        if not remote_log and run_name:
            try:
                found = ssh.run(
                    f"ls -t {self._REMOTE_PROJECT_DIR}/logs/{run_name}*.log "
                    "2>/dev/null | head -1",
                    timeout=10).strip()
                remote_log = found or self._REMOTE_FALLBACK_LOG
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
            self._append_local_log(
                f"{self._LOG_PREFIX} Downloading final logs from GPU instance...")
            self._fetch_log_lines(ssh, remote_log, final=True)
            self._append_local_log(
                f"{self._LOG_PREFIX} Downloading checkpoints from GPU instance...")
            self._download_checkpoints(ssh, run_name, instance_id)
            self._append_local_log(f"{self._LOG_PREFIX} Download complete.")
            with self._lock:
                if self._instance_id == instance_id:
                    self._on_job_ended()

        # 6. nvidia-smi — only when instance alive and enough time has passed
        if time.time() - self._last_status_t >= self._STATUS_MIN_SECS:
            self._refresh_status_cache(ssh)
            self._last_status_t = time.time()

    def _append_local_log(self, message: str) -> None:
        """Append a status message to the local log file."""
        with open(self._local_log, "a") as f:
            f.write(message + "\n")

    def _fetch_log_lines(self, ssh: InstanceSSH, remote_log: str,
                         final: bool = False) -> None:
        """Fetch log lines since _log_offset via SSH; append to local log."""
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

    # -- Remote file helpers -------------------------------------------------

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

    def _ensure_remote_checkpoint(self, local_checkpoint: str,
                                  remote_dir: str | None = None) -> str:
        """Ensure a checkpoint file (and its .json companion) exists on the
        remote instance. Returns the remote path to use for --checkpoint."""
        if remote_dir is None:
            remote_dir = f"{self._REMOTE_PROJECT_DIR}/output"
        chk = Path(local_checkpoint)
        remote_path = f"{remote_dir}/{chk.name}"
        if not self._remote_file_exists(remote_path):
            self._ensure_remote_dir(remote_dir)
            self._upload_file(str(chk), remote_path)
            meta = chk.with_suffix(".json")
            if meta.exists():
                self._upload_file(str(meta), f"{remote_dir}/{meta.name}")
        return remote_path

    # -- Checkpoint selection helpers ----------------------------------------

    def _latest_timestamped(self, msgpack_files: list[str]) -> str | None:
        """Return the newest timestamped .msgpack path, or None if there are none."""
        stamped = sorted(f for f in msgpack_files if "_latest." not in f)
        return stamped[-1] if stamped else None

    def _latest_rolling(self, msgpack_files: list[str]) -> str | None:
        """Return the _latest.msgpack path, or None if absent."""
        return next((f for f in msgpack_files if "_latest." in f), None)

    def _pick_checkpoint_stem(self, msgpack_files: list[str]) -> str | None:
        """Decide which checkpoint to download and return its stem.

        Rules:
        - If a timestamped checkpoint is newer than (or equal to) _latest,
          training completed normally — use the timestamped one.
        - If _latest is newer than all timestamped files, or no timestamped
          files exist, training ended early — use _latest.
        """
        stamped = self._latest_timestamped(msgpack_files)
        rolling = self._latest_rolling(msgpack_files)
        if not stamped and not rolling:
            return None
        if not stamped:
            return Path(rolling).stem
        if not rolling:
            return Path(stamped).stem
        # Both exist — compare mtimes
        try:
            if self._get_mtime(rolling) > self._get_mtime(stamped):
                return Path(rolling).stem  # _latest is newer — training didn't finish
        except Exception:
            pass  # mtime fetch failed — fall through to timestamped
        return Path(stamped).stem

    # -- Session restore -----------------------------------------------------

    def _restore_from_state(self, entry: dict) -> "SocketJobHandle | None":
        """Generic cloud-slot restore path.

        1. Verify instance alive via _check_instance_alive
        2. Rebuild SSH via _build_ssh_from_info
        3. Probe pgrep for the saved run_name
        4. _attach sets slot fields; if pgrep showed dead, state goes
           directly to "downloading" so the next poll tick runs the final
           log + checkpoint download path.
        Returns a handle on success, None if the instance is gone or restore
        failed at any step.
        """
        instance_id = entry.get("instance_id")
        run_name    = entry.get("run_name")
        if not instance_id or not run_name:
            return None

        try:
            info = self._check_instance_alive(str(instance_id))
        except Exception as e:
            log_debug("restore: instance verify failed (transient)",
                      instance_id=instance_id, error=str(e), gpu_index=self._index)
            return None
        if info is None:
            log_debug("restore: instance gone", instance_id=instance_id,
                      gpu_index=self._index)
            return None

        try:
            ssh = self._build_ssh_from_info(info)
        except Exception as e:
            log_error("restore: SSH build failed", instance_id=instance_id,
                      error=str(e), gpu_index=self._index)
            return None

        try:
            training_alive = bool(_ssh_find_run(ssh, run_name, timeout=15).strip())
        except Exception as e:
            # SSH check failed — uncertain. Assume alive so the normal poll
            # loop decides; better than prematurely entering download mode.
            log_debug("restore: pgrep probe failed, assuming alive",
                      instance_id=instance_id, error=str(e), gpu_index=self._index)
            training_alive = True

        log_debug("restore: probe result", instance_id=instance_id,
                  run_name=run_name, training_alive=training_alive,
                  gpu_index=self._index)

        try:
            self._attach(
                str(instance_id), ssh, info, run_name,
                training_status="running" if training_alive else "downloading",
            )
        except Exception as e:
            log_error("restore: _attach failed",
                      instance_id=instance_id, run_name=run_name,
                      error=str(e), gpu_index=self._index)
            return None

        with self._lock:
            if self._job_alive():
                return self._make_handle()
        return None

    # -- Status command ------------------------------------------------------

    def _cmd_status(self, conn: socket.socket) -> None:
        """Return combined instance + GPU status over the socket."""
        try:
            with self._lock:
                payload = {
                    "instance_id":    self._instance_id,
                    "instance_state": self._instance_state,
                    "cost_per_hour":  self._cost_per_hour,
                    "total_cost":     self._total_cost,
                }
            payload.update(self._gpu_stats())
            conn.sendall((json.dumps(payload) + "\n").encode())
        except Exception as e:
            conn.sendall((json.dumps({"error": str(e)}) + "\n").encode())
        finally:
            conn.close()

    def _download_checkpoints(self, ssh: InstanceSSH, run_name: str,
                              instance_id: str) -> None:
        """Download checkpoint files (.msgpack + .json) from the remote instance."""
        if not run_name:
            return
        local_output = self._backend.project_root / "output"
        local_output.mkdir(exist_ok=True)
        try:
            remote_output = f"{self._REMOTE_PROJECT_DIR}/output"
            listing = ssh.run(
                f"ls {remote_output}/{run_name}*.msgpack "
                f"{remote_output}/{run_name}*.json 2>/dev/null || true",
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
            log_debug("nvidia-smi SSH fetch failed",
                      instance_id=self._instance_id, error=str(e))

    def _try_rediscover_training(self, ssh: InstanceSSH,
                                  instance_id: str) -> None:
        """Scan the instance for an out-of-band train.py --run-name process.

        Called from _poll when the slot is attached but marked idle. If a
        matching process is found, the slot is promoted back to "running"
        with _run_name extracted from the process args. Silent on SSH
        errors — we'll retry next poll.
        """
        discovered = _ssh_scan_train_run_name(ssh, instance_id)
        if discovered is None:
            return

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
            self._log_offset      = 0
            self._log_complete    = False
        log_debug("rediscovered train.py process",
                  instance_id=instance_id, run_name=discovered)

    # -- Hardware-specific commands -----------------------------------------

    def _gpu_stats(self) -> dict:
        with self._status_lock:
            return dict(self._status_cache)

    def _cmd_submit(self, conn, json_str: str) -> None:
        try:
            cfg      = json.loads(json_str)
            run_name = cfg["run_name"]
            script   = cfg["script"]
            params   = cfg.get("params", {})
            chk      = cfg.get("checkpoint")
            log_name = Path(cfg.get("log_file", f"{run_name}.log")).name
            remote_log = f"{self._REMOTE_PROJECT_DIR}/logs/{log_name}"

            with self._lock:
                instance_id = self._instance_id
                ssh         = self._ssh

            if not instance_id:
                raise RuntimeError("No instance attached to this slot — attach one first.")
            if not ssh:
                raise RuntimeError("No SSH connection to instance.")

            # Reject if a training job for this run_name is already tracked
            with self._lock:
                already_tracked = (self._training_status in ("submitted", "running")
                                   and self._run_name == run_name)
            if already_tracked:
                raise RuntimeError(
                    f"A training job for '{run_name}' is already running on this "
                    "instance. Kill it first."
                )
            try:
                existing = _ssh_find_run(ssh, run_name)
                if existing.strip():
                    pid = existing.strip().split()[0]
                    raise RuntimeError(
                        f"Instance already has a running '{run_name}' process "
                        f"(PID {pid}). Kill it first."
                    )
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"Could not check for existing processes: {e}")

            # Stage server state before launch so a timeout leaves clean state
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

            # Copy training files via SCP
            ssh.run(f"mkdir -p {self._REMOTE_PROJECT_DIR}/logs", timeout=10)
            project_root = self._backend.project_root
            for fname in ("train.py", "policy.py", "default_params.json"):
                src = project_root / fname
                if src.exists():
                    ssh.copy_to(str(src), f"{self._REMOTE_PROJECT_DIR}/{fname}")

            # Ensure checkpoint is available on the instance
            if chk:
                chk = self._ensure_remote_checkpoint(
                    chk, remote_dir=f"{self._REMOTE_PROJECT_DIR}/output")

            # Build training command and launch via SSH (background + disown)
            python_prefix = " ".join(self.python_cmd)
            cmd = (f"cd {self._REMOTE_PROJECT_DIR} && "
                   f"{python_prefix} {script} --run-name {run_name}")
            for key, val in params.items():
                cmd += f" --{key.replace('_', '-')} {val}"
            if chk:
                cmd += f" --checkpoint {chk}"
            cmd += f" > {remote_log} 2>&1 </dev/null & disown"
            _ssh_execute_async(ssh, cmd, instance_id=instance_id)
            # Poll thread confirms the process started and transitions to "running"
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
            ssh         = self._ssh
        try:
            if not instance_id:
                conn.sendall(b"error no job running\n")
                return
            if ssh is None:
                conn.sendall(b"error no SSH connection\n")
                return
            _ssh_execute_async(
                ssh, f"pkill -f 'train.py.*--run-name {run_name}' || true",
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
# _BaseCloudBackend — shared backend lifecycle for SSH-based cloud providers
# ---------------------------------------------------------------------------

class _BaseCloudBackend:
    """Manages a pool of virtual GPU slots backed by cloud instances.

    Subclasses set _SLOT_CLASS and implement _list_running_instances. All
    attach/restart/poll lifecycle is shared here. Pure server — use
    BackendClient from compute_backend_client.py to interact.
    """

    _SLOT_CLASS: type = _BaseCloudGPUServer
    _INSTANCE_POLL_INTERVAL: int = 30

    def __init__(self,
                 num_slots:    int         = 1,
                 project_root: Path | None = None,
                 sock_path:    str         = "",
                 backend_id:   str         = ""):
        self._pre_init()
        self.project_root = project_root or Path.cwd()
        self._servers: dict[int, _BaseCloudGPUServer] = {
            i: self._SLOT_CLASS(i, self) for i in range(num_slots)
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
                         name=f"{backend_id}-instance-poll").start()

    # -- Provider-specific hooks --------------------------------------------

    def _pre_init(self) -> None:
        """Optional provider-specific setup before slots are created."""
        pass

    def _list_running_instances(self) -> list[dict]:
        """Return provider info dicts for currently-running instances."""
        raise NotImplementedError

    # -- Shared lifecycle ---------------------------------------------------

    def _poll_instances_loop(self) -> None:
        """Background thread: discover running instances on a fixed interval
        and attach any untracked ones to idle slots."""
        while True:
            time.sleep(self._INSTANCE_POLL_INTERVAL)
            try:
                self._attach_untracked_instances()
            except Exception as e:
                log_error("_attach_untracked_instances failed", exc=e)

    def _attach_untracked_instances(self) -> None:
        """Discover running cloud instances not yet tracked and attach them to
        idle slots. Alive checking and log/status fetching are handled by each
        slot's _poll() thread."""
        running = self._list_running_instances()

        tracked = set()
        for server in self._servers.values():
            with server._lock:
                if server._instance_id:
                    tracked.add(str(server._instance_id))

        for info in running:
            instance_id = str(info["id"])
            if instance_id in tracked:
                continue

            # Attach to an idle slot, or create a new one
            idle = next(
                (s for s in self._servers.values() if s._instance_id is None),
                None,
            )
            if idle is None:
                slot_id = max(self._servers) + 1
                idle = self._SLOT_CLASS(slot_id, self)
                idle.start()
                self._servers[slot_id] = idle

            # Build SSH connection — may fail (e.g. runtime not ready on RunPod)
            try:
                ssh = idle._build_ssh_from_info(info)
            except Exception as e:
                log_debug("Could not build SSH for instance",
                          instance_id=instance_id, error=str(e))
                continue

            # Scan for an existing train.py --run-name process. Only treat
            # the slot as "running a job" if we can extract a real run_name
            # from the process args — never fall back to the provider's
            # auto-generated name, which would poison checkpoint paths.
            discovered_run_name = _ssh_scan_train_run_name(ssh, instance_id)
            idle._attach(
                instance_id, ssh, info, discovered_run_name,
                training_status="running" if discovered_run_name else "idle",
            )


# ---------------------------------------------------------------------------
# _BackendServer
# ---------------------------------------------------------------------------

class _BackendServer:
    """Daemon thread exposing a single backend socket for discovery and state queries."""

    def __init__(self, servers: "dict[int, _BaseGPUServer]",
                 sock_path: str, backend_id: str):
        self._servers    = servers
        self._sock_path  = sock_path
        self._backend_id = backend_id
        self._ready      = threading.Event()

    def start(self) -> None:
        t = threading.Thread(target=self._serve, daemon=True, name="backend-server")
        t.start()
        self._ready.wait(timeout=5.0)

    def _serve(self) -> None:
        try:
            os.unlink(self._sock_path)
        except FileNotFoundError:
            pass
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self._sock_path)
        srv.listen(16)
        self._ready.set()
        try:
            while True:
                conn, _ = srv.accept()
                threading.Thread(target=self._handle_client, args=(conn,),
                                 daemon=True).start()
        finally:
            srv.close()
            try:
                os.unlink(self._sock_path)
            except FileNotFoundError:
                pass

    def _handle_client(self, conn: socket.socket) -> None:
        try:
            cmd = _read_line(conn)
            if cmd == "info":
                self._cmd_info(conn)
            elif cmd == "discover":
                self._cmd_discover(conn)
            elif cmd == "running_jobs":
                self._cmd_running_jobs(conn)
            elif cmd.startswith("register_existing "):
                self._cmd_register_existing(conn, cmd[18:])
            elif cmd.startswith("reattach "):
                self._cmd_reattach(conn, cmd[9:])
            elif cmd.startswith("query_job "):
                self._cmd_query_job(conn, cmd[10:])
            else:
                conn.sendall(f"error unknown: {cmd}\n".encode())
                conn.close()
        except Exception as e:
            log_error("_BackendServer._handle_client unhandled exception",
                      exc=e, cmd=locals().get("cmd", "?"))
            # Always send a response so the client doesn't see an empty read
            # and blow up in json.loads("").
            try:
                conn.sendall(f"error server exception: {e}\n".encode())
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

    def _cmd_info(self, conn: socket.socket) -> None:
        conn.sendall((json.dumps({
            "backend_id": self._backend_id,
            "gpu_count":  len(self._servers),
            "host":       socket.gethostname(),
        }) + "\n").encode())
        conn.close()

    def _cmd_discover(self, conn: socket.socket) -> None:
        gpus = [
            {
                "index":      s._index,
                "name":       s.name,
                "sock_path":  s.sock_path,
                "backend_id": self._backend_id,
            }
            for s in sorted(self._servers.values(), key=lambda s: s._index)
        ]
        conn.sendall((json.dumps(gpus) + "\n").encode())
        conn.close()

    def _cmd_running_jobs(self, conn: socket.socket) -> None:
        jobs = []
        for s in self._servers.values():
            h = s.current_handle()
            if h is not None:
                jobs.append(_handle_to_dict(h, s.sock_path, self._backend_id))
        conn.sendall((json.dumps(jobs) + "\n").encode())
        conn.close()

    def _cmd_register_existing(self, conn: socket.socket, json_str: str) -> None:
        try:
            d      = json.loads(json_str)
            server = self._servers.get(d["gpu_id"])
            if server is None:
                conn.sendall(b"error gpu not found\n")
                return
            handle = server.register_job(d["run_name"], d["job_id"], d["log_file"])
            if handle is None:
                conn.sendall(b"error job not alive\n")
                return
            conn.sendall((json.dumps(
                _handle_to_dict(handle, server.sock_path, self._backend_id)
            ) + "\n").encode())
        except Exception as e:
            conn.sendall(f"error {e}\n".encode())
        finally:
            conn.close()

    def _cmd_reattach(self, conn: socket.socket, json_str: str) -> None:
        """Restore a slot from a runs_state.json entry persisted by the
        dashboard. Looks for a slot already holding this instance_id (in case
        _attach_untracked_instances ran first and attached it as idle); if
        none, falls back to the saved gpu_id. Returns the restored handle,
        or null if the instance is gone / restore failed.
        """
        try:
            entry       = json.loads(json_str)
            instance_id = entry.get("instance_id")
            gpu_id      = entry.get("gpu_id")
            target = None
            if instance_id is not None:
                for server in self._servers.values():
                    with server._lock:
                        if str(server._instance_id) == str(instance_id):
                            target = server
                            break
            if target is None and gpu_id is not None:
                target = self._servers.get(int(gpu_id))
            if target is None:
                conn.sendall(b"null\n")
                return
            handle = target._restore_from_state(entry)
            if handle is None:
                conn.sendall(b"null\n")
                return
            conn.sendall((json.dumps(
                _handle_to_dict(handle, target.sock_path, self._backend_id)
            ) + "\n").encode())
        except Exception as e:
            log_error("_cmd_reattach failed", exc=e)
            conn.sendall(f"error {e}\n".encode())
        finally:
            conn.close()

    def _cmd_query_job(self, conn: socket.socket, gpu_id_str: str) -> None:
        try:
            server = self._servers.get(int(gpu_id_str))
            if server is None:
                conn.sendall(b"null\n")
                return
            h = server.current_handle()
            if h is None:
                conn.sendall(b"null\n")
                return
            conn.sendall((json.dumps(
                _handle_to_dict(h, server.sock_path, self._backend_id)
            ) + "\n").encode())
        except Exception as e:
            conn.sendall(f"error {e}\n".encode())
        finally:
            conn.close()


