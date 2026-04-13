"""Common server-side infrastructure shared across compute backends.

_BaseGPUServer  — socket loop and protocol skeleton. Backends subclass this
                  and implement the four hardware-specific commands plus the
                  two job-state helpers (_job_alive, _job_id). Cloud slots
                  additionally override _build_ssh_from_info and
                  _apply_restored_state so the reattach path can rebuild
                  their state after a dashboard restart. Lock invariant:
                  _job_alive(), _job_id(), and _make_handle() must always
                  be called with self._lock held.

_BackendServer  — backend-level socket server for discovery, running-job
                  queries, and the dashboard-driven `reattach` command.
                  Works with any dict[int, _BaseGPUServer].

State persistence lives on the dashboard side (runs_state.json). Backends
are stateless on disk; they rebuild slot state on each restart via
_attach_untracked_instances (discovery) and _cmd_reattach (dashboard replay).
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

from ripe_grm.compute_backend_client import SocketJobHandle
from ripe_grm.dashboard_log import log_debug, log_error


# ---------------------------------------------------------------------------
# _BaseGPUServer
# ---------------------------------------------------------------------------

class _BaseGPUServer(ABC):
    """Per-GPU socket server skeleton.

    Subclasses implement the four hardware-specific commands
    (_cmd_status, _cmd_submit, _cmd_cancel, _cmd_logs) and the three
    job-state helpers (register_job, _job_alive, _job_id).
    """

    # Instance states
    INSTANCE_NONE    = "none"      # no instance attached to this slot
    INSTANCE_RUNNING = "running"   # instance alive and reachable
    INSTANCE_OFFLINE = "offline"   # instance not responding (transient or gone)

    _POLL_INTERVAL: int = 0        # 0 = no background polling; backends set non-zero
    _ABSENT_THRESHOLD: int = 2     # consecutive failed heartbeats before detaching

    def __init__(self, index: int, name: str, sock_path: str):
        self._index    = index
        self.name      = name
        self.sock_path = sock_path
        self._lock     = threading.Lock()
        self._run_name: str | None = None
        self._log_file: str = ""
        self._ready    = threading.Event()
        # -- Instance state (cloud backends) ---
        self._instance_id:    str | None = None
        self._instance_state: str = self.INSTANCE_NONE
        self._absent_polls:   int = 0
        self._cost_per_hour:  float = 0.0
        self._total_cost:     float = 0.0

    @property
    def python_cmd(self) -> list[str]:
        """Command prefix to invoke Python for a training job.
        Returns [sys.executable, "-u"] by default; cloud backends override this."""
        return [sys.executable, "-u"]

    def _discover_extra(self) -> dict:
        """Extra fields to include in the discover response for this GPU.
        Override in backends to expose cost_per_hour, instance_id, etc."""
        return {}

    # -- Instance heartbeat (cloud backends) ----------------------------------

    def _check_instance_alive(self, instance_id: str) -> dict | None:
        """Query the cloud provider to check if the instance is alive.
        Return instance info dict if alive, None if gone or unreachable.
        Cloud backends must override this."""
        return None

    def _on_heartbeat_alive(self, info: dict) -> None:
        """Called (with self._lock held) when heartbeat confirms instance alive.
        Override to update backend-specific fields (cost, etc.) from info."""
        pass

    def _on_instance_detached(self) -> None:
        """Clear all instance state — slot becomes fully idle.
        Called (with self._lock held) when instance is confirmed gone.
        Cloud backends should override to clear backend-specific state."""
        self._on_job_ended()
        self._instance_id    = None
        self._instance_state = self.INSTANCE_NONE
        self._absent_polls   = 0
        self._cost_per_hour  = 0.0
        self._total_cost     = 0.0

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

    # -- Checkpoint selection helpers (shared across cloud backends) ----------

    def _latest_timestamped(self, msgpack_files: list[str]) -> str | None:
        """Return the newest timestamped .msgpack path, or None if there are none."""
        stamped = sorted(f for f in msgpack_files if "_latest." not in f)
        return stamped[-1] if stamped else None

    def _latest_rolling(self, msgpack_files: list[str]) -> str | None:
        """Return the _latest.msgpack path, or None if absent."""
        return next((f for f in msgpack_files if "_latest." in f), None)

    def _get_mtime(self, path: str) -> int:
        """Return mtime (epoch seconds) for a remote/local checkpoint file.
        Subclasses must override this for their storage backend."""
        raise NotImplementedError

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

    # -- Remote file helpers (cloud backends override) -----------------------

    def _remote_file_exists(self, remote_path: str) -> bool:
        """Check if a file exists on the remote instance.
        Cloud backends must override this."""
        raise NotImplementedError

    def _upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a local file to the remote instance.
        Cloud backends must override this."""
        raise NotImplementedError

    def _ensure_remote_dir(self, remote_dir: str) -> None:
        """Create a directory on the remote instance if needed.
        Cloud backends must override this."""
        raise NotImplementedError

    def _ensure_remote_checkpoint(self, local_checkpoint: str,
                                  remote_dir: str = "/root/project/output"
                                  ) -> str:
        """Ensure a checkpoint file (and its .json companion) exists on the
        remote instance. Returns the remote path to use for --checkpoint."""
        chk = Path(local_checkpoint)
        remote_path = f"{remote_dir}/{chk.name}"
        if not self._remote_file_exists(remote_path):
            self._ensure_remote_dir(remote_dir)
            self._upload_file(str(chk), remote_path)
            meta = chk.with_suffix(".json")
            if meta.exists():
                self._upload_file(str(meta), f"{remote_dir}/{meta.name}")
        return remote_path

    def start(self) -> None:
        t = threading.Thread(target=self._serve, daemon=True,
                             name=f"gpu-server-{self._index}")
        t.start()
        self._ready.wait(timeout=5.0)
        if self._POLL_INTERVAL > 0:
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

    def _poll(self) -> None:
        """Periodic background work (alive check, log fetch, status).
        No-op by default — override in backends that need it."""
        pass

    # -- Abstract interface --------------------------------------------------

    def register_job(self, run_name: str, job_id: str,
                     log_file: str) -> "SocketJobHandle | None":
        """Re-attach to an already-running job by opaque job id.

        Used by the `register_existing` socket command, which is driven by
        dashboard._attach_running_processes — a scan of local nvidia-smi
        compute-apps. Only the local backend overrides this; cloud backends
        never see a local PID for a remote pod, so the default no-op applies.
        Dashboard-restart reattach for cloud slots goes through `_cmd_reattach`
        / `_restore_from_state` instead.
        """
        return None

    # -- Session restore (cloud backends) ------------------------------------
    #
    # Called when the dashboard restarts and replays saved runs_state.json
    # entries via _cmd_reattach. Each cloud slot verifies the instance still
    # exists, rebuilds SSH, probes whether the training process is alive, and
    # either resumes normal polling ("running") or jumps straight into the
    # log+checkpoint download path ("downloading") so the run can finalize
    # cleanly even though we missed its actual exit.

    def _build_ssh_from_info(self, info: dict):
        """Rebuild a backend-specific SSH connection from provider instance info.
        Cloud backends override. Local backend does not use this."""
        raise NotImplementedError

    def _apply_restored_state(self, instance_id: str, ssh, run_name: str,
                              info: dict, entry: dict,
                              training_alive: bool) -> None:
        """Write restored state onto the slot. Called by _restore_from_state
        with the SSH-probed liveness verdict. Cloud backends override to set
        provider-specific fields (cost, remote_log, etc). MUST acquire
        self._lock internally."""
        raise NotImplementedError

    def _restore_from_state(self, entry: dict) -> "SocketJobHandle | None":
        """Generic cloud-slot restore path.

        1. Verify instance alive via _check_instance_alive
        2. Rebuild SSH via _build_ssh_from_info
        3. Probe pgrep for the saved run_name
        4. _apply_restored_state sets slot fields; if pgrep showed dead,
           state goes directly to "downloading" so the next poll tick runs
           the final log + checkpoint download path.
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

        training_alive = False
        try:
            out = ssh.run(
                f"pgrep -fa 'train.py' | grep -E -- '--run-name {run_name}( |$)' || true",
                timeout=15)
            training_alive = bool(out.strip())
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
            self._apply_restored_state(
                str(instance_id), ssh, run_name, info, entry, training_alive,
            )
        except Exception as e:
            log_error("restore: _apply_restored_state failed",
                      instance_id=instance_id, run_name=run_name,
                      error=str(e), gpu_index=self._index)
            return None

        with self._lock:
            if self._job_alive():
                return self._make_handle()
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

    @staticmethod
    def _read_line(conn: socket.socket) -> str:
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(256)
            if not chunk:
                return ""
            buf += chunk
        return buf.split(b"\n", 1)[0].decode().strip()

    def _handle_client(self, conn: socket.socket) -> None:
        try:
            cmd = self._read_line(conn)
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

    @staticmethod
    def _read_line(conn: socket.socket) -> str:
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(256)
            if not chunk:
                return ""
            buf += chunk
        return buf.split(b"\n", 1)[0].decode().strip()

    def _handle_client(self, conn: socket.socket) -> None:
        try:
            cmd = self._read_line(conn)
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
            log_error("_BackendServer._handle_client unhandled exception", exc=e)
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
                **s._discover_extra(),
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
                jobs.append({
                    "run_name":   h.run_name,
                    "id":         h.id,
                    "gpu_id":     h.gpu_id,
                    "log_file":   h.log_file,
                    "sock_path":  s.sock_path,
                    "backend_id": self._backend_id,
                })
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
            conn.sendall((json.dumps({
                "run_name":   handle.run_name,
                "id":         handle.id,
                "gpu_id":     handle.gpu_id,
                "log_file":   handle.log_file,
                "sock_path":  server.sock_path,
                "backend_id": self._backend_id,
            }) + "\n").encode())
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
            conn.sendall((json.dumps({
                "run_name":   handle.run_name,
                "id":         handle.id,
                "gpu_id":     handle.gpu_id,
                "log_file":   handle.log_file,
                "sock_path":  target.sock_path,
                "backend_id": self._backend_id,
            }) + "\n").encode())
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
            conn.sendall((json.dumps({
                "run_name":   h.run_name,
                "id":         h.id,
                "gpu_id":     h.gpu_id,
                "log_file":   h.log_file,
                "sock_path":  server.sock_path,
                "backend_id": self._backend_id,
            }) + "\n").encode())
        except Exception as e:
            conn.sendall(f"error {e}\n".encode())
        finally:
            conn.close()


