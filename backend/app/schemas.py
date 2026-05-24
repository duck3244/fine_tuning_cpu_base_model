from pydantic import BaseModel, Field
from typing import Optional, Literal


class TrainRequest(BaseModel):
    dataset_id: str
    model_name: str = "skt/kogpt2-base-v2"
    max_steps: int = Field(100, ge=1, le=10000)
    batch_size: int = Field(1, ge=1, le=32)
    learning_rate: float = Field(5e-5, gt=0)
    epochs: int = Field(1, ge=1)


class JobInfo(BaseModel):
    job_id: str
    status: Literal["running", "completed", "failed", "stopped"]
    started_at: str
    finished_at: Optional[str] = None
    exit_code: Optional[int] = None
    output_dir: Optional[str] = None
    request: Optional[TrainRequest] = None


class InferRequest(BaseModel):
    model_id: str
    question: str
    max_new_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.8, gt=0, le=2.0)


class InferResponse(BaseModel):
    question: str
    answer: str
    elapsed_ms: int


class DatasetInfo(BaseModel):
    dataset_id: str
    filename: str
    rows: int
    size_bytes: int


class ModelInfo(BaseModel):
    model_id: str
    path: str
    created_at: str
    size_bytes: int
