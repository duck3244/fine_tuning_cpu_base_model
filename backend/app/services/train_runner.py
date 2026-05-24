import os
import signal
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path

from app.config import BACKEND_ROOT, DATASETS_DIR, MODELS_DIR, LOGS_DIR
from app.schemas import TrainRequest
from app.services.job_store import job_store


def _watch_process(job_id: str, proc: subprocess.Popen, output_dir: Path):
    exit_code = proc.wait()
    status = "completed" if exit_code == 0 else ("stopped" if exit_code < 0 else "failed")
    job_store.update(
        job_id,
        status=status,
        exit_code=exit_code,
        finished_at=datetime.utcnow().isoformat(),
        output_dir=str(output_dir) if output_dir.exists() else None,
    )


def start_training(req: TrainRequest) -> str:
    csv_path = DATASETS_DIR / f"{req.dataset_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {req.dataset_id}")

    job_id = uuid.uuid4().hex[:12]
    output_dir = MODELS_DIR / job_id
    log_path = LOGS_DIR / f"{job_id}.log"

    job = job_store.create(job_id, req.model_dump())

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": ""}
    cmd = [
        "python", "-u", "main_train_cpu.py",
        "--csv_path", str(csv_path),
        "--output_dir", str(output_dir),
        "--model_name", req.model_name,
        "--batch_size", str(req.batch_size),
        "--epochs", str(req.epochs),
        "--max_steps", str(req.max_steps),
        "--learning_rate", str(req.learning_rate),
    ]

    log_file = open(log_path, "wb")
    proc = subprocess.Popen(
        cmd,
        cwd=str(BACKEND_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    job_store.update(job_id, pid=proc.pid, log_path=str(log_path))
    threading.Thread(target=_watch_process, args=(job_id, proc, output_dir), daemon=True).start()
    return job_id


def stop_training(job_id: str) -> bool:
    job = job_store.get(job_id)
    if not job or job["status"] != "running" or not job.get("pid"):
        return False
    try:
        os.killpg(os.getpgid(job["pid"]), signal.SIGTERM)
        return True
    except ProcessLookupError:
        return False
