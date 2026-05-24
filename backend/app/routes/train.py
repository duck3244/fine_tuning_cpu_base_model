import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas import JobInfo, TrainRequest
from app.services.job_store import job_store
from app.services.train_runner import start_training, stop_training

router = APIRouter(prefix="/api/train", tags=["train"])


@router.post("", response_model=JobInfo)
def start(req: TrainRequest):
    try:
        job_id = start_training(req)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return JobInfo(**job_store.get(job_id))


@router.get("/jobs", response_model=list[JobInfo])
def list_jobs():
    return [JobInfo(**j) for j in job_store.list()]


@router.get("/jobs/{job_id}", response_model=JobInfo)
def get_job(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JobInfo(**job)


@router.post("/jobs/{job_id}/stop")
def stop(job_id: str):
    if not stop_training(job_id):
        raise HTTPException(400, "Job not running or not found")
    return {"stopped": job_id}


@router.get("/jobs/{job_id}/stream")
async def stream_log(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    log_path = Path(job["log_path"]) if job.get("log_path") else None
    if not log_path or not log_path.exists():
        raise HTTPException(404, "Log not available")

    async def event_gen():
        with open(log_path, "rb") as f:
            while True:
                line = f.readline()
                if line:
                    text = line.decode("utf-8", errors="replace").rstrip("\n")
                    yield f"data: {text}\n\n"
                else:
                    current = job_store.get(job_id)
                    if not current or current["status"] != "running":
                        yield f"event: end\ndata: {current['status'] if current else 'unknown'}\n\n"
                        return
                    await asyncio.sleep(0.5)

    return StreamingResponse(event_gen(), media_type="text/event-stream")
