# Ripe AutoTrain

Ripe AutoTrain is a GPU management and training automation app for managing and monitoring reinforcement learning training runs.

## Installation

```bash
pip install -e /path/to/ripe_autotrain
```

## Usage

Run from your training project directory:

```bash
cd /path/to/your/training/project
ripe_autotrain
```

The dashboard reads configuration from the current working directory.

---

## Configuration files

Both files live in your **training project directory** (where you invoke `ripe_autotrain`).

### `compute_registry.json`

Defines the GPU backends the dashboard will connect to. At least one entry is required.

**Local GPUs:**

```json
[
  {
    "sock_path":  "/tmp/drl_backend.sock",
    "backend_id": "local",
    "script":     "local_backend.py"
  }
]
```

**Vast.ai cloud GPUs:**

```json
[
  {
    "sock_path":   "/tmp/drl_vast_backend.sock",
    "backend_id":  "vast",
    "script":      "vastai_backend.py",
    "num_slots":   2,
    "offer_query": "num_gpus=1 gpu_name=RTX_4090 rentable=true verified=true",
    "image":       "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    "disk_gb":     20
  }
]
```

**Mixed (local + cloud):**

```json
[
  {
    "sock_path":  "/tmp/drl_backend.sock",
    "backend_id": "local",
    "script":     "local_backend.py"
  },
  {
    "sock_path":   "/tmp/drl_vast_backend.sock",
    "backend_id":  "vast",
    "script":      "vastai_backend.py",
    "num_slots":   1,
    "offer_query": "num_gpus=1 gpu_name=RTX_4090 rentable=true verified=true",
    "image":       "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    "disk_gb":     20
  }
]
```

Fields:

| Field | Required | Description |
|---|---|---|
| `sock_path` | yes | Unix socket path for IPC |
| `backend_id` | yes | Unique identifier for this backend |
| `script` | yes | Backend script name (`local_backend.py` or `vastai_backend.py`) |
| `num_slots` | Vast only | Number of concurrent cloud instances |
| `offer_query` | Vast only | Vast.ai search filter string |
| `image` | Vast only | Docker image for the remote instance |
| `disk_gb` | Vast only | Disk size to provision on the remote instance |

---

### `dashboard_config.json`

Tells the dashboard where your training script saves checkpoints and logs. This file is optional — if absent, the dashboard scans the entire current directory recursively and points TensorBoard at `.`.

```json
{
  "output_dir": "output",
  "log_dir":    "logs/humanoid"
}
```

Fields:

| Field | Description |
|---|---|
| `output_dir` | Directory where `.msgpack` checkpoints and `.json` metadata are saved |
| `log_dir` | Directory where TensorBoard event files are written |

Both paths are relative to the training project directory.

---

### `default_params.json`

Optional. Defines the hyperparameter fields shown in the spawn/queue dialogs, along with their default values. Keys must match the argument names accepted by your training script.

```json
{
  "num_envs":    4096,
  "lr":          3e-4,
  "gamma":       0.99,
  "clip_eps":    0.2,
  "epochs":      4,
  "batch_size":  2048,
  "total_steps": 50000000
}
```
