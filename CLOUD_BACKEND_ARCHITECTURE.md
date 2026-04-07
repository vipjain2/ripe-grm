# Cloud GPU Backend Architecture

This document describes the architecture of the compute backend system well enough to implement a new backend for a different cloud GPU provider (e.g. RunPod).

---

## Overview

The backend system decouples the training dashboard from the underlying compute provider. Each backend runs as a **separate OS process** and exposes a **Unix domain socket** interface. The dashboard communicates with backends exclusively through these sockets using a simple line-delimited text protocol.

```
Dashboard process                Backend process (e.g. vastai_backend.py)
─────────────────                ─────────────────────────────────────────
BackendClient  ──── backend.sock ──── _BackendServer
_GPUClient     ──── gpu_N.sock   ──── _VastGPUServer (one per slot)
```

---

## Files

| File | Role |
|------|------|
| `compute_backend_client.py` | Dashboard-side: `BackendClient`, `_GPUClient`, `SocketJobHandle`, `GPU`, `JobHandle` ABCs |
| `compute_backend_server.py` | Shared server infrastructure: `_BaseGPUServer`, `_BackendServer`, `_BaseBackend` |
| `local_backend.py` | Local NVIDIA GPU implementation |
| `vastai_backend.py` | Vast.ai cloud implementation |
| `compute_registry.json` | Config file listing all backends to start |
| `dashboard_backend.py` | Dashboard startup: reads registry, launches backend processes, connects clients |

---

## Startup Flow

1. Dashboard calls `init_backends()` in `dashboard_backend.py`.
2. `init_backends` reads `compute_registry.json` (list of backend entries).
3. For each entry, it kills any existing process for that script and relaunches it via `subprocess.Popen`.
4. It then waits (up to 10s) for each backend socket to become responsive.
5. It calls `BackendClient.discover_gpus()` on each backend to get the list of available GPU slots.

### `compute_registry.json` format

```json
[
  {
    "sock_path":   "/tmp/drl_backend.sock",
    "backend_id":  "local",
    "script":      "local_backend.py"
  },
  {
    "sock_path":   "/tmp/drl_vast_backend.sock",
    "backend_id":  "vast",
    "script":      "vastai_backend.py",
    "num_slots":   2
  }
]
```

Each backend script is invoked as:
```
python <script> <sock_path> <backend_id>
```

---

## Socket Protocol

All sockets use Unix domain stream sockets (`AF_UNIX`, `SOCK_STREAM`). Messages are line-delimited (`\n`). All responses are a single JSON line followed by `\n`, except log streams which are raw lines.

### Backend socket (one per backend process)

| Command | Response | Description |
|---------|----------|-------------|
| `info` | `{"backend_id": ..., "gpu_count": ..., "host": ...}` | Backend metadata |
| `discover` | `[{"index":0, "name":..., "cost_per_hour":..., "sock_path":..., "backend_id":..., ...}, ...]` | List all GPU slots |
| `running_jobs` | `[{"run_name":..., "id":..., "gpu_id":..., "log_file":..., "sock_path":..., "backend_id":...}, ...]` | All currently running jobs |
| `register_existing <json>` | `{"run_name":..., ...}` or `error ...` | Re-attach to a job after restart |
| `query_job <gpu_id>` | `{"run_name":..., ...}` or `null` | Query current job on a specific GPU |

### GPU socket (one per GPU slot)

| Command | Response | Description |
|---------|----------|-------------|
| `info` | `{"index":..., "name":..., "cost_per_hour":..., "sock_path":...}` | GPU metadata |
| `status` | `{"util_pct":..., "temp_c":..., "mem_used_mb":..., "mem_total_mb":...}` | GPU utilization snapshot |
| `job_status` | `{"run_name":..., "id":..., "running": true/false}` | Is a job currently running? |
| `submit <json>` | `ok <job_id>` or `error <msg>` | Launch a training job |
| `cancel` | `ok` or `error <msg>` | Kill the running job |
| `logs` | raw log lines (streaming until job ends) | Stream the job's stdout/stderr |

The `submit` JSON payload:
```json
{
  "run_name":   "exp_001",
  "script":     "train.py",
  "params":     {"lr": 3e-4, "num_steps": 100000},
  "checkpoint": "/path/to/checkpoint.msgpack",
  "log_file":   "/tmp/drl_runs/exp_001_20240101_120000.log"
}
```

---

## Server-Side Class Hierarchy

### `_BaseGPUServer` (compute_backend_server.py)

Abstract base class for a single GPU slot server. Handles the socket accept loop and dispatches commands. Backends subclass this and implement the four abstract commands plus three job-state helpers.

**State managed by base class:**
- `_lock` — threading lock; job-state helpers and `_on_job_ended` must be called with lock held
- `_run_name`, `_log_file` — current job metadata
- `_ready` — event signaled when socket is listening

**Abstract methods to implement:**

```python
def register_job(self, run_name: str, job_id: str, log_file: str) -> SocketJobHandle | None:
    """Re-attach to an already-running job after a dashboard restart.
    Must verify the job is actually alive. Return None if it is not."""

def _job_alive(self) -> bool:
    """True if the current job is still running. Called with self._lock held."""

def _job_id(self) -> str:
    """Opaque job identifier (PID, instance ID, etc.). Called with self._lock held."""

def _cmd_status(self, conn: socket.socket) -> None:
    """Return a JSON GPUStatus dict: util_pct, temp_c, mem_used_mb, mem_total_mb."""

def _cmd_submit(self, conn: socket.socket, json_str: str) -> None:
    """Launch a training job. Send 'ok <job_id>\\n' on success, 'error <msg>\\n' on failure."""

def _cmd_cancel(self, conn: socket.socket) -> None:
    """Kill the running job. Send 'ok\\n' or 'error <msg>\\n'."""

def _cmd_logs(self, conn: socket.socket) -> None:
    """Stream job log lines to conn until the job ends, then close conn."""
```

**Optional overrides:**

```python
def _on_job_ended(self) -> None:
    """Called with self._lock held when a job is detected as ended.
    Clear subclass job state (e.g. _pid, _process, _instance_id)."""

def _discover_extra(self) -> dict:
    """Extra fields merged into the discover response for this GPU slot.
    Use to expose cost_per_hour, instance_id, total_cost, etc."""
```

**Background poll thread:**

Set `_POLL_INTERVAL = N` (seconds) in the subclass to enable a background poll thread. The base class starts a daemon thread that calls `_poll()` every N seconds. Override `_poll()` for non-blocking background work (alive checks, log fetching, status updates).

```python
_POLL_INTERVAL = 10   # set non-zero to enable

def _poll(self) -> None:
    """Called every _POLL_INTERVAL seconds in a background thread."""
```

### `_BackendServer` (compute_backend_server.py)

Handles the backend-level socket (one per backend process). Constructed with a `dict[int, _BaseGPUServer]` and a `save_state` callback. No subclassing needed.

### `_BaseBackend` (compute_backend_server.py)

State persistence mixin. Implements `_save_state()` and `_restore_state()` using a JSON file. The concrete backend must assign `self._servers` and `self._state_file` before calling `_restore_state()`.

---

## Client-Side Classes

### `BackendClient` (compute_backend_client.py)

Connects to the backend socket. Main methods:
- `info()` — backend metadata
- `discover_gpus()` → `list[GPU]` — returns `_GPUClient` instances
- `running_jobs()` → `list[SocketJobHandle]`
- `register_existing(run_name, gpu_id, log_file, job_id)` → `SocketJobHandle | None`

### `_GPUClient` (compute_backend_client.py)

Thin wrapper around a GPU socket. Implements the `GPU` ABC. Attributes populated from `discover`:
- `index`, `name`, `backend_id`
- `cost_per_hour` — 0.0 for local, USD/hr for cloud
- `instance_id` — cloud instance ID, `None` for local
- `total_cost` — cumulative cost computed in the backend

### `SocketJobHandle` (compute_backend_client.py)

Tracks a running job. Holds `run_name`, `gpu_id`, `log_file`, `id` (opaque job ID), `sock_path`. Methods:
- `is_running()` — sends `job_status` to the GPU socket
- `cancel()` — sends `cancel`
- `open_log_reader()` → `queue.Queue[str | None]` — connects and streams `logs`, puts `None` at end

---

## Implementing a New Cloud Backend

To implement a backend for a new provider (e.g. RunPod), create a new file `runpod_backend.py` following this pattern:

### 1. Per-slot server class

```python
class _RunPodGPUServer(_BaseGPUServer):
    _POLL_INTERVAL = 10  # enable background poll thread

    def __init__(self, slot_id: int, backend: "RunPodBackend"):
        super().__init__(slot_id, f"RunPod Slot {slot_id}", f"/tmp/drl_runpod_{slot_id}.sock")
        self._backend      = backend
        self._pod_id       = None    # cloud instance identifier
        self._ssh_host     = None
        self._ssh_port     = None
        self._remote_log   = None
        self._cost_per_hour = 0.0
        self._total_cost   = 0.0
        self._pod_alive    = False
        self._absent_polls = 0       # consecutive API misses before marking dead
        self._training_alive   = False
        self._absent_training  = 0   # consecutive pgrep misses before marking dead
        self._local_log    = f"/tmp/drl_runpod_{slot_id}.log"
        self._log_offset   = 0
        self._log_complete = False
        self._status_cache = {"util_pct": 0, "temp_c": 0, "mem_used_mb": 0, "mem_total_mb": 0}
        self._status_lock  = threading.Lock()
        self._last_status_t = 0.0
```

### 2. Job-state helpers (called with lock held)

```python
    def _job_alive(self) -> bool:
        return self._pod_alive and self._training_alive

    def _job_id(self) -> str:
        return str(self._pod_id) if self._pod_id else "-"

    def _on_job_ended(self) -> None:
        self._pod_id = self._ssh_host = self._ssh_port = self._remote_log = None
        self._cost_per_hour = self._total_cost = 0.0
        self._pod_alive = self._training_alive = False
        self._absent_polls = self._absent_training = 0
        self._log_offset = 0
        self._log_complete = True  # unblock any waiting _cmd_logs
        with self._status_lock:
            self._status_cache = {"util_pct": 0, "temp_c": 0, "mem_used_mb": 0, "mem_total_mb": 0}
```

### 3. Session restore

```python
    def register_job(self, run_name, job_id, log_file):
        """Called at dashboard restart. Verify the pod and training process are alive."""
        info = _get_pod(job_id)           # provider API call
        if not info or not _pod_running(info):
            return None
        ssh_host, ssh_port = _ssh_info(info)
        try:
            out = _ssh(ssh_host, ssh_port,
                       f"pgrep -fa 'train.py' | grep -- '--run-name {run_name}'",
                       timeout=15)
            if not out.strip():
                return None             # instance alive but training stopped
        except Exception:
            return None                 # SSH failed — be conservative on restore
        with self._lock:
            self._pod_id = job_id
            self._ssh_host = ssh_host
            self._ssh_port = ssh_port
            self._remote_log = log_file
            self._run_name = run_name
            self._log_file = self._local_log
            self._pod_alive = True
            self._training_alive = True
            # ... set cost fields from info
        return self._make_handle()
```

### 4. Background poll (`_poll`)

The poll thread is the only place that makes blocking API or SSH calls. Everything else reads cached state.

```python
    def _poll(self) -> None:
        with self._lock:
            pod_id = self._pod_id
            ssh_host = self._ssh_host
            ssh_port = self._ssh_port
            remote_log = self._remote_log
            run_name = self._run_name or ""
            was_pod_alive = self._pod_alive
            was_train_alive = self._training_alive

        if not pod_id:
            return

        # 1. Pod alive check via provider API (never SSH for alive check)
        info = _get_pod(pod_id)
        is_alive = bool(info and _pod_running(info))

        with self._lock:
            if self._pod_id == pod_id:
                if is_alive:
                    self._absent_polls = 0
                    self._pod_alive = True
                    self._total_cost = _compute_cost(info)   # dph * uptime / 60
                else:
                    self._absent_polls += 1
                    if self._absent_polls >= 2:              # require 2 misses
                        self._pod_alive = False

        if not (ssh_host and ssh_port) or not is_alive:
            return

        # 2. Training process check via SSH pgrep
        try:
            out = _ssh(ssh_host, ssh_port,
                       f"pgrep -fa 'train.py' | grep -- '--run-name {run_name}'",
                       timeout=10)
            process_found = bool(out.strip())
        except Exception:
            process_found = True         # SSH error on known-running job — keep state

        with self._lock:
            if self._pod_id == pod_id:
                if process_found:
                    self._absent_training = 0
                    self._training_alive = True
                else:
                    self._absent_training += 1
                    if self._absent_training >= 2:           # require 2 misses
                        self._training_alive = False

        is_training_alive = process_found or self._absent_training < 2

        # 3. Discover remote log path if not yet known
        if not remote_log and run_name:
            try:
                found = _ssh(ssh_host, ssh_port,
                             f"ls -t /root/project/logs/{run_name}*.log 2>/dev/null | head -1",
                             timeout=10).strip()
                if found:
                    with self._lock:
                        if self._pod_id == pod_id:
                            self._remote_log = found
                    remote_log = found
            except Exception:
                pass

        if not remote_log:
            return

        # 4. Incremental log fetch
        if is_training_alive:
            self._fetch_log_lines(ssh_host, ssh_port, remote_log)

        # 5. Final log download when training just ended
        if was_train_alive and not is_training_alive:
            self._fetch_log_lines(ssh_host, ssh_port, remote_log, final=True)
            self._log_complete = True

        # 6. Final download if instance died before training ended
        if was_pod_alive and not is_alive and self._absent_polls >= 2 and not self._log_complete:
            self._fetch_log_lines(ssh_host, ssh_port, remote_log, final=True)
            self._log_complete = True

        # 7. GPU status via SSH nvidia-smi (throttled)
        if time.time() - self._last_status_t >= 30:
            self._refresh_status_cache(ssh_host, ssh_port)
            self._last_status_t = time.time()
```

### 5. Commands

**`_cmd_status`** — return cached status (no blocking calls):
```python
    def _cmd_status(self, conn) -> None:
        with self._status_lock:
            payload = dict(self._status_cache)
        conn.sendall((json.dumps(payload) + "\n").encode())
        conn.close()
```

**`_cmd_submit`** — assume instance already rented and SSH-accessible:
```python
    def _cmd_submit(self, conn, json_str) -> None:
        cfg = json.loads(json_str)
        # 1. rsync project files to remote
        # 2. SSH to launch: nohup python -u train.py --run-name ... > remote_log 2>&1 &
        # 3. Update self._run_name, self._remote_log, self._training_alive, etc.
        # 4. Call self._backend._save_state()
        conn.sendall(f"ok {instance_id}\n".encode())
```

**`_cmd_cancel`** — call provider destroy API:
```python
    def _cmd_cancel(self, conn) -> None:
        _destroy_pod(self._pod_id)    # provider API call
        conn.sendall(b"ok\n")
        conn.close()
```

**`_cmd_logs`** — read local file written by poll thread:
```python
    def _cmd_logs(self, conn) -> None:
        # Wait for local log file to appear, then tail it until self._log_complete
        # The poll thread writes to self._local_log incrementally
        ...
```

### 6. Backend class

```python
class RunPodBackend(_BaseBackend):
    def __init__(self, num_slots=1, project_root=None, state_file=None,
                 sock_path="/tmp/drl_runpod_backend.sock", backend_id="runpod"):
        self.project_root = project_root or Path.cwd()
        self._state_file  = state_file or (self.project_root / "runpod_backend_state.json")
        self._servers = {i: _RunPodGPUServer(i, self) for i in range(num_slots)}
        for s in self._servers.values():
            s.start()
        self._backend_server = _BackendServer(self._servers, self._save_state,
                                              sock_path, backend_id)
        self._backend_server.start()
        self._restore_state()
        # Optional: background thread to discover untracked pods
        threading.Thread(target=self._poll_pods_loop, daemon=True).start()

    def _poll_pods_loop(self) -> None:
        """Discover running pods not yet tracked and attach them to idle slots."""
        while True:
            try:
                self._attach_untracked_pods()
            except Exception:
                pass
            time.sleep(30)

    def _attach_untracked_pods(self) -> None:
        pods = _list_running_pods()            # provider API
        tracked = {s._pod_id for s in self._servers.values() if s._pod_id}
        for pod in pods:
            if str(pod["id"]) in tracked:
                continue
            ssh_host, ssh_port = _ssh_info(pod)
            # SSH to detect training process — skip (don't assume alive) on failure
            try:
                ps_out = _ssh(ssh_host, ssh_port,
                              "ps -eo args | grep 'train.py' | grep -v grep | head -1",
                              timeout=10)
                if not ps_out.strip():
                    continue             # pod alive but no training — skip
                run_name = _extract_run_name(ps_out)
            except Exception:
                continue                 # SSH failed during discovery — retry next cycle
            # Attach to an idle slot
            idle = next((s for s in self._servers.values() if s._pod_id is None), None)
            if idle is None:
                slot_id = max(self._servers) + 1
                idle = _RunPodGPUServer(slot_id, self)
                idle.start()
                self._servers[slot_id] = idle
            with idle._lock:
                idle._pod_id = str(pod["id"])
                idle._ssh_host = ssh_host
                idle._ssh_port = ssh_port
                idle._run_name = run_name
                idle._log_file = idle._local_log
                idle._remote_log = ""    # poll thread will discover path
                idle._pod_alive = True
                idle._training_alive = True
```

### 7. Entry point

```python
if __name__ == "__main__":
    sock_path  = sys.argv[1] if len(sys.argv) > 1 else "/tmp/drl_runpod_backend.sock"
    backend_id = sys.argv[2] if len(sys.argv) > 2 else "runpod"
    RunPodBackend(sock_path=sock_path, backend_id=backend_id)
    print(f"RunPod backend ready  id={backend_id}  sock={sock_path}", flush=True)
    signal.pause()
```

### 8. Register in `compute_registry.json`

```json
{
  "sock_path":   "/tmp/drl_runpod_backend.sock",
  "backend_id":  "runpod",
  "script":      "runpod_backend.py",
  "num_slots":   2
}
```

---

## Key Design Principles

**Non-blocking main thread.** All SSH and provider API calls happen in the background poll thread. Socket command handlers (`_cmd_status`, `_cmd_logs`, etc.) return cached state immediately. Socket connections have a 5–15s timeout to prevent the dashboard from blocking.

**Instance alive ≠ job alive.** Cloud instances can be running with no training process. `_job_alive()` must return `True` only when both the instance is alive AND the training process is running. Use separate flags (`_pod_alive`, `_training_alive`) updated independently.

**Fault tolerance via counters.** Require 2 consecutive API/SSH misses before marking a job as dead (`_absent_polls >= 2`, `_absent_training >= 2`). A single timeout should not kill a tracked job.

**Conservative discovery.** In `_attach_untracked_pods`, if SSH fails, skip and retry next cycle. Never assume a newly discovered instance has a running training job without positive confirmation.

**Local log files.** The poll thread downloads log lines incrementally from the remote instance and appends them to a local file in `/tmp`. `_cmd_logs` reads this local file and signals completion via `_log_complete`. This avoids a live SSH connection in the log-streaming hot path.

**State persistence.** `_BaseBackend._save_state()` writes running job metadata (run_name, job_id, gpu_id, log_file) to a JSON file. On restart, `_restore_state()` calls `register_job()` for each entry, which verifies liveness before re-attaching.
