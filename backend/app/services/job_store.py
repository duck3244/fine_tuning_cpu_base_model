import threading
from datetime import datetime
from typing import Dict, Optional


class JobStore:
    """In-memory job state. Single-user MVP — no persistence."""

    def __init__(self):
        self._jobs: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str, request: dict) -> dict:
        with self._lock:
            if any(j["status"] == "running" for j in self._jobs.values()):
                raise RuntimeError("Another training job is already running")
            job = {
                "job_id": job_id,
                "status": "running",
                "started_at": datetime.utcnow().isoformat(),
                "finished_at": None,
                "exit_code": None,
                "output_dir": None,
                "request": request,
                "pid": None,
                "log_path": None,
            }
            self._jobs[job_id] = job
            return job

    def get(self, job_id: str) -> Optional[dict]:
        return self._jobs.get(job_id)

    def list(self) -> list:
        return sorted(self._jobs.values(), key=lambda j: j["started_at"], reverse=True)

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)

    def running_job(self) -> Optional[dict]:
        for j in self._jobs.values():
            if j["status"] == "running":
                return j
        return None


job_store = JobStore()
