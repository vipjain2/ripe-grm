"""Compute backend abstraction — client-side only.

Common client classes used by the dashboard to communicate with backend
processes over Unix domain sockets. Backend implementations (e.g.
local_backend.py) live in separate modules.

Usage:
    client = BackendClient("/tmp/drl_backend.sock")
    gpus   = client.discover_gpus()   # list[GPU]

    for gpu in gpus:
        print(gpu.index, gpu.name, gpu.cost_per_hour, gpu.status())

    handle = gpu.submit(JobConfig(...))
    q      = handle.open_log_reader()
"""

from __future__ import annotations

import json
import queue
import socket
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# GPU status snapshot
# ---------------------------------------------------------------------------

@dataclass
class GPUStatus:
    util_pct:     int    # 0–100
    temp_c:       int    # Celsius
    mem_used_mb:  int
    mem_total_mb: int

    @property
    def mem_free_mb(self) -> int:
        return self.mem_total_mb - self.mem_used_mb

    @property
    def is_idle(self) -> bool:
        """Heuristic: <500 MiB used and <5% utilisation."""
        return self.mem_used_mb < 500 and self.util_pct < 5


# ---------------------------------------------------------------------------
# Job config and handle
# ---------------------------------------------------------------------------

@dataclass
class JobConfig:
    run_name:   str
    script:     str                              # e.g. "train.py", relative to project root
    params:     dict[str, object] = field(default_factory=dict)
    checkpoint: str | None = None
    log_file:   str = ""


class JobHandle(ABC):
    """Tracks a job occupying a GPU slot."""

    run_name: str
    gpu_id:   int
    log_file: str
    id:       str   # opaque; PID for LocalBackend, instance ID for cloud

    @abstractmethod
    def is_running(self) -> bool: ...

    @abstractmethod
    def cancel(self) -> None: ...

    @abstractmethod
    def open_log_reader(self) -> "queue.Queue[str | None]":
        """Connect to the job's log stream. Returns a queue of lines.
        None sentinel is put when the job ends."""
        ...


# ---------------------------------------------------------------------------
# GPU — identity, cost, status, dispatch
# ---------------------------------------------------------------------------

class GPU(ABC):
    index:         int
    name:          str
    cost_per_hour: float   # USD; 0.0 for local hardware

    @abstractmethod
    def status(self) -> GPUStatus: ...

    @abstractmethod
    def submit(self, config: JobConfig) -> JobHandle: ...

    @abstractmethod
    def env_vars(self) -> dict[str, str]:
        """Env vars needed to pin a subprocess to this GPU."""
        ...


# ---------------------------------------------------------------------------
# SocketJobHandle — communicates with a GPU server via Unix socket
# ---------------------------------------------------------------------------

class SocketJobHandle(JobHandle):

    def __init__(self, run_name: str, gpu_id: int, log_file: str,
                 id: str, sock_path: str, backend_id: str = "local"):
        self.run_name   = run_name
        self.gpu_id     = gpu_id
        self.log_file   = log_file
        self.id         = id
        self.backend_id = backend_id
        self._sock_path = sock_path

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect(self._sock_path)
        return sock

    def _request(self, cmd: str) -> str:
        sock = self._connect()
        sock.sendall((cmd + "\n").encode())
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
        sock.close()
        return buf.decode().strip()

    def is_running(self) -> bool:
        try:
            resp = self._request("job_status")
            return json.loads(resp).get("running", False)
        except Exception:
            return False

    def cancel(self) -> None:
        try:
            self._request("cancel")
        except Exception:
            pass

    def open_log_reader(self) -> "queue.Queue[str | None]":
        q: queue.Queue = queue.Queue()
        sock_path = self._sock_path

        def _read() -> None:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(sock_path)
                sock.sendall(b"logs\n")
            except Exception:
                q.put(None)
                return
            buf = ""
            try:
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        q.put(None)
                        break
                    buf += chunk.decode("utf-8", errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        q.put(line)
            except Exception:
                q.put(None)
            finally:
                sock.close()

        threading.Thread(target=_read, daemon=True).start()
        return q


# ---------------------------------------------------------------------------
# _GPUClient — thin GPU wrapper that sends commands to a _GPUServer socket
# ---------------------------------------------------------------------------

class _GPUClient(GPU):
    cost_per_hour = 0.0

    def __init__(self, index: int, name: str, sock_path: str, backend_id: str = "local"):
        self.index      = index
        self.backend_id = backend_id
        self.name       = f"GPU({backend_id}, {index})"
        self._sock_path = sock_path

    def _request(self, cmd: str, timeout: float = 15.0) -> str:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(self._sock_path)
        sock.sendall((cmd + "\n").encode())
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
        sock.close()
        return buf.decode().strip()

    def status(self) -> GPUStatus:
        d = json.loads(self._request("status"))
        if "error" in d:
            raise RuntimeError(d["error"])
        return GPUStatus(**d)

    def env_vars(self) -> dict[str, str]:
        return {"CUDA_VISIBLE_DEVICES": str(self.index)}

    def submit(self, config: JobConfig) -> SocketJobHandle:
        payload = json.dumps({
            "run_name":   config.run_name,
            "script":     config.script,
            "params":     config.params,
            "checkpoint": config.checkpoint,
            "log_file":   config.log_file,
        })
        resp = self._request(f"submit {payload}")
        if not resp.startswith("ok "):
            raise RuntimeError(f"submit failed: {resp}")
        return SocketJobHandle(
            run_name  = config.run_name,
            gpu_id    = self.index,
            log_file  = config.log_file,
            id        = resp[3:],
            sock_path = self._sock_path,
        )


# ---------------------------------------------------------------------------
# BackendClient — queries the backend socket; entry point for the dashboard
# ---------------------------------------------------------------------------

class BackendClient:
    """Thin client for the backend socket. Use discover_gpus() to get GPUs."""

    def __init__(self, sock_path: str):
        self._sock_path = sock_path

    def _request(self, cmd: str) -> str:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect(self._sock_path)
        sock.sendall((cmd + "\n").encode())
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
        sock.close()
        return buf.decode().strip()

    def info(self) -> dict:
        """Return backend metadata: backend_id, gpu_count, host."""
        return json.loads(self._request("info"))

    def discover_gpus(self) -> list[GPU]:
        """Query the backend socket for all available GPUs."""
        gpus = json.loads(self._request("discover"))
        return [_GPUClient(g["index"], g["name"], g["sock_path"], g["backend_id"]) for g in gpus]

    def running_jobs(self) -> list[SocketJobHandle]:
        jobs = json.loads(self._request("running_jobs"))
        return [
            SocketJobHandle(j["run_name"], j["gpu_id"], j["log_file"],
                            j["id"], j["sock_path"], j["backend_id"])
            for j in jobs
        ]

    def register_existing(self, run_name: str, gpu_id: int,
                          log_file: str, job_id: str) -> "SocketJobHandle | None":
        payload = json.dumps({"run_name": run_name, "gpu_id": gpu_id,
                              "log_file": log_file, "job_id": job_id})
        resp = self._request(f"register_existing {payload}")
        if resp.startswith("error") or resp == "null":
            return None
        d = json.loads(resp)
        return SocketJobHandle(d["run_name"], d["gpu_id"], d["log_file"],
                               d["id"], d["sock_path"], d["backend_id"])

    def query_job(self, gpu_id: int) -> "SocketJobHandle | None":
        resp = self._request(f"query_job {gpu_id}")
        if resp == "null" or resp.startswith("error"):
            return None
        d = json.loads(resp)
        return SocketJobHandle(d["run_name"], d["gpu_id"], d["log_file"],
                               d["id"], d["sock_path"], d["backend_id"])

