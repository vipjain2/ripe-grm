"""Local GPU backend server.

Implements _GPUServer and LocalBackend for machines with local NVIDIA GPUs.
Common server infrastructure lives in compute_backend_server.py.

Run as a standalone process (started by dashboard via compute_registry.json):
    python local_backend.py <sock_path> <backend_id>
    python local_backend.py /tmp/drl_backend.sock local
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

DEFAULT_BACKEND_SOCK = "/tmp/drl_backend.sock"


# ---------------------------------------------------------------------------
# _GPUServer — local NVIDIA GPU implementation
# ---------------------------------------------------------------------------

class _GPUServer(_BaseGPUServer):
    """Local GPU server using nvidia-smi for status and subprocess for job dispatch."""

    def __init__(self, index: int, name: str, backend: "LocalBackend"):
        super().__init__(index, name, f"/tmp/drl_gpu_{index}.sock")
        self._backend  = backend
        self._process: subprocess.Popen | None = None
        self._pid:      int | None = None

    # -- Job-state helpers (called with self._lock held) ---------------------

    def _job_alive(self) -> bool:
        if self._process is not None:
            return self._process.poll() is None
        if self._pid:
            try:
                os.kill(self._pid, 0)
                return True
            except (OSError, ProcessLookupError):
                return False
        return False

    def _job_id(self) -> str:
        return str(self._pid) if self._pid else "-"

    def _on_job_ended(self) -> None:
        self._pid     = None
        self._process = None

    # -- Session restore -----------------------------------------------------

    def register_job(self, run_name: str, job_id: str,
                     log_file: str) -> "SocketJobHandle | None":
        try:
            pid = int(job_id)
            os.kill(pid, 0)
        except (ValueError, OSError, ProcessLookupError):
            return None
        with self._lock:
            self._pid      = pid
            self._process  = None
            self._run_name = run_name
            self._log_file = log_file
        return self._make_handle()

    # -- Hardware-specific commands -----------------------------------------

    def _cmd_status(self, conn) -> None:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi", f"--id={self._index}",
                    "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True, stderr=subprocess.DEVNULL,
            )
            util, temp, mem_used, mem_total = [x.strip() for x in out.strip().split(",")]
            conn.sendall((json.dumps({
                "util_pct":     int(util),
                "temp_c":       int(temp),
                "mem_used_mb":  int(mem_used),
                "mem_total_mb": int(mem_total),
            }) + "\n").encode())
        except Exception as e:
            conn.sendall((json.dumps({"error": str(e)}) + "\n").encode())
        finally:
            conn.close()

    def _cmd_submit(self, conn, json_str: str) -> None:
        try:
            cfg = json.loads(json_str)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self._index)

            cmd = [sys.executable, "-u", cfg["script"],
                   "--run-name", cfg["run_name"]]
            for key, val in cfg.get("params", {}).items():
                cmd += [f"--{key.replace('_', '-')}", str(val)]
            if cfg.get("checkpoint"):
                cmd += ["--checkpoint", cfg["checkpoint"]]

            log_path = Path(cfg.get("log_file", ""))
            log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(log_path, "w") as lf:
                proc = subprocess.Popen(
                    cmd,
                    stdout=lf, stderr=subprocess.STDOUT,
                    start_new_session=True, env=env,
                    cwd=self._backend.project_root,
                )

            with self._lock:
                self._process  = proc
                self._pid      = proc.pid
                self._run_name = cfg["run_name"]
                self._log_file = str(log_path)

            self._backend._save_state()
            conn.sendall(f"ok {proc.pid}\n".encode())
        except Exception as e:
            conn.sendall(f"error {e}\n".encode())
        finally:
            conn.close()

    def _cmd_cancel(self, conn) -> None:
        with self._lock:
            proc = self._process
            pid  = self._pid
        try:
            if proc and proc.poll() is None:
                proc.terminate()
                conn.sendall(b"ok\n")
            elif pid:
                os.kill(pid, 15)
                conn.sendall(b"ok\n")
            else:
                conn.sendall(b"error no job running\n")
        except (OSError, ProcessLookupError) as e:
            conn.sendall(f"error {e}\n".encode())
        finally:
            conn.close()

    def _cmd_logs(self, conn) -> None:
        with self._lock:
            log_file = self._log_file
        if not log_file:
            conn.close()
            return
        try:
            while not Path(log_file).exists():
                with self._lock:
                    if not self._job_alive():
                        return
                time.sleep(0.1)
            with open(log_file) as f:
                while True:
                    line = f.readline()
                    if line:
                        conn.sendall(line.encode())
                    else:
                        with self._lock:
                            if not self._job_alive():
                                break
                        time.sleep(0.05)
        except Exception:
            pass
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# LocalBackend
# ---------------------------------------------------------------------------

class LocalBackend(_BaseBackend):
    """Starts per-GPU socket servers and the backend socket.
    Pure server — use BackendClient from compute_backend_client.py to interact."""

    def __init__(self, project_root: Path | None = None,
                 state_file: Path | None = None,
                 sock_path: str = DEFAULT_BACKEND_SOCK,
                 backend_id: str = "local"):
        self.project_root = project_root or Path.cwd()
        self._state_file  = state_file or (self.project_root / "backend_state.json")
        self._servers: dict[int, _GPUServer] = {}
        self._init_gpu_servers()
        self._backend_server = _BackendServer(self._servers, self._save_state,
                                              sock_path, backend_id)
        self._backend_server.start()
        self._restore_state()

    def _init_gpu_servers(self) -> None:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name",
                 "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return
        for line in out.strip().splitlines():
            if not line.strip():
                continue
            idx_str, name = line.split(",", 1)
            idx = int(idx_str.strip())
            server = _GPUServer(idx, name.strip(), self)
            server.start()
            self._servers[idx] = server


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _sock_path  = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BACKEND_SOCK
    _backend_id = sys.argv[2] if len(sys.argv) > 2 else "local"

    LocalBackend(sock_path=_sock_path, backend_id=_backend_id)
    print(f"Local backend ready  id={_backend_id}  sock={_sock_path}", flush=True)
    signal.pause()
