"""Common server-side infrastructure shared across compute backends.

_BaseGPUServer  — socket loop and protocol skeleton. Backends subclass this
                  and implement the four hardware-specific commands plus three
                  job-state helpers. Lock invariant: _job_alive(), _job_id(),
                  and _make_handle() must always be called with self._lock held.

_BackendServer  — backend-level socket server for discovery and state queries.
                  Works with any dict[int, _BaseGPUServer].

_BaseBackend    — state persistence mixin (save/restore running job list).
                  Concrete backends must set self._servers and self._state_file
                  before calling _restore_state().
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

from ripe_autotrain.compute_backend_client import SocketJobHandle


# ---------------------------------------------------------------------------
# _BaseGPUServer
# ---------------------------------------------------------------------------

class _BaseGPUServer(ABC):
    """Per-GPU socket server skeleton.

    Subclasses implement the four hardware-specific commands
    (_cmd_status, _cmd_submit, _cmd_cancel, _cmd_logs) and the three
    job-state helpers (register_job, _job_alive, _job_id).
    """

    def __init__(self, index: int, name: str, sock_path: str):
        self._index    = index
        self.name      = name
        self.sock_path = sock_path
        self._lock     = threading.Lock()
        self._run_name: str | None = None
        self._log_file: str = ""
        self._ready    = threading.Event()

    _POLL_INTERVAL: int = 0  # 0 = no background polling; backends set non-zero

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
        except Exception:
            pass
        while True:
            time.sleep(self._POLL_INTERVAL)
            try:
                self._poll()
            except Exception:
                pass

    def _poll(self) -> None:
        """Periodic background work (alive check, log fetch, status).
        No-op by default — override in backends that need it."""
        pass

    # -- Abstract interface --------------------------------------------------

    @abstractmethod
    def register_job(self, run_name: str, job_id: str,
                     log_file: str) -> "SocketJobHandle | None":
        """Re-attach to an already-running job. Returns None if the job is gone."""
        ...

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

    @abstractmethod
    def _cmd_status(self, conn: socket.socket) -> None: ...

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
        except Exception:
            try:
                conn.close()
            except Exception:
                pass

    def _cmd_info(self, conn: socket.socket) -> None:
        conn.sendall((json.dumps({
            "index":         self._index,
            "name":          self.name,
            "cost_per_hour": 0.0,
            "sock_path":     self.sock_path,
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

    def __init__(self, servers: "dict[int, _BaseGPUServer]", save_state,
                 sock_path: str, backend_id: str):
        self._servers    = servers
        self._save_state = save_state
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
            elif cmd.startswith("query_job "):
                self._cmd_query_job(conn, cmd[10:])
            else:
                conn.sendall(f"error unknown: {cmd}\n".encode())
                conn.close()
        except Exception:
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
                "index":         s._index,
                "name":          s.name,
                "cost_per_hour": 0.0,
                "sock_path":     s.sock_path,
                "backend_id":    self._backend_id,
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
        self._save_state()
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
            self._save_state()
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


# ---------------------------------------------------------------------------
# _BaseBackend
# ---------------------------------------------------------------------------

class _BaseBackend:
    """State persistence mixin for backends.

    Concrete backends must assign self._servers (dict[int, _BaseGPUServer])
    and self._state_file (Path) before calling _restore_state().
    """

    _servers:    "dict[int, _BaseGPUServer]"
    _state_file: Path

    def _save_state(self) -> None:
        entries = []
        for server in self._servers.values():
            h = server.current_handle()
            if h is not None:
                entries.append({
                    "run_name": h.run_name,
                    "id":       h.id,
                    "gpu_id":   h.gpu_id,
                    "log_file": h.log_file,
                })
        self._state_file.write_text(json.dumps(entries, indent=2))

    def _restore_state(self) -> None:
        if not self._state_file.exists():
            return
        try:
            entries = json.loads(self._state_file.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for entry in entries:
            server = self._servers.get(entry.get("gpu_id"))
            if server is None:
                continue
            job_id = str(entry.get("id") or entry.get("pid", ""))
            server.register_job(
                run_name = entry["run_name"],
                job_id   = job_id,
                log_file = entry.get("log_file", ""),
            )
        self._save_state()
