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

from ripe_grm.compute_backend_client import SocketJobHandle
from ripe_grm.compute_backend_server import _BackendServer, _BaseGPUServer
from ripe_grm.dashboard_log import log_debug, log_error

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

    def _gpu_stats(self) -> dict:
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
            return {
                "util_pct":     int(util),
                "temp_c":       int(temp),
                "mem_used_mb":  int(mem_used),
                "mem_total_mb": int(mem_total),
            }
        except Exception as e:
            log_error("_gpu_stats nvidia-smi failed", exc=e, gpu_index=self._index)
            return {"util_pct": 0, "temp_c": 0, "mem_used_mb": 0, "mem_total_mb": 0}

    def _cmd_submit(self, conn, json_str: str) -> None:
        try:
            cfg = json.loads(json_str)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self._index)

            cmd = [*self.python_cmd, cfg["script"],
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

            conn.sendall(f"ok {proc.pid}\n".encode())
        except Exception as e:
            log_error("_cmd_submit failed", exc=e, gpu_index=self._index)
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
            log_error("_cmd_cancel failed", exc=e, gpu_index=self._index, pid=pid)
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
        except Exception as e:
            log_debug("_cmd_logs stream ended with error", error=str(e),
                      log_file=log_file)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# LocalBackend
# ---------------------------------------------------------------------------

class LocalBackend:
    """Starts per-GPU socket servers and the backend socket.
    Pure server — use BackendClient from compute_backend_client.py to interact."""

    def __init__(self, project_root: Path | None = None,
                 sock_path: str = DEFAULT_BACKEND_SOCK,
                 backend_id: str = "local"):
        self.project_root = project_root or Path.cwd()
        self._servers: dict[int, _GPUServer] = {}
        self._init_gpu_servers()
        self._backend_server = _BackendServer(self._servers, sock_path, backend_id)
        self._backend_server.start()
        self._discover_running_jobs()

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

    def _discover_running_jobs(self) -> None:
        """Scan nvidia-smi for running compute processes and re-attach any train.py jobs."""
        try:
            uuid_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,gpu_uuid",
                 "--format=csv,noheader"],
                text=True, stderr=subprocess.DEVNULL,
            )
        except Exception:
            return
        uuid_to_idx: dict[str, int] = {}
        for line in uuid_out.strip().splitlines():
            if not line.strip():
                continue
            idx_str, uuid = [x.strip() for x in line.split(",", 1)]
            uuid_to_idx[uuid] = int(idx_str)

        try:
            apps_out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                 "--format=csv,noheader"],
                text=True, stderr=subprocess.DEVNULL,
            )
        except Exception:
            return

        for line in apps_out.strip().splitlines():
            if not line.strip():
                continue
            parts = [x.strip() for x in line.split(",", 1)]
            if len(parts) != 2:
                continue
            pid_str, uuid = parts
            try:
                pid = int(pid_str)
            except ValueError:
                continue
            gpu_idx = uuid_to_idx.get(uuid)
            if gpu_idx is None:
                continue
            server = self._servers.get(gpu_idx)
            if server is None:
                continue

            try:
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
                args = cmdline.decode("utf-8", errors="replace").split("\x00")
            except OSError:
                continue

            if not any("train.py" in a for a in args):
                continue

            run_name = None
            log_file = ""
            for i, arg in enumerate(args):
                if arg == "--run-name" and i + 1 < len(args):
                    run_name = args[i + 1]
                elif arg == "--log-file" and i + 1 < len(args):
                    log_file = args[i + 1]

            if run_name is None:
                continue

            server.register_job(run_name=run_name, job_id=str(pid), log_file=log_file)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _sock_path  = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BACKEND_SOCK
    _backend_id = sys.argv[2] if len(sys.argv) > 2 else "local"

    LocalBackend(sock_path=_sock_path, backend_id=_backend_id)
    print(f"Local backend ready  id={_backend_id}  sock={_sock_path}", flush=True)
    signal.pause()
