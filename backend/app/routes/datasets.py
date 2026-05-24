import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import DATASETS_DIR
from app.schemas import DatasetInfo

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("", response_model=list[DatasetInfo])
def list_datasets():
    out = []
    for p in sorted(DATASETS_DIR.glob("*.csv")):
        try:
            rows = sum(1 for _ in open(p, encoding="utf-8")) - 1
        except Exception:
            rows = 0
        out.append(DatasetInfo(
            dataset_id=p.stem,
            filename=p.name,
            rows=max(rows, 0),
            size_bytes=p.stat().st_size,
        ))
    return out


@router.post("/upload", response_model=DatasetInfo)
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files accepted")

    dataset_id = uuid.uuid4().hex[:12]
    target = DATASETS_DIR / f"{dataset_id}.csv"
    target.write_bytes(await file.read())

    try:
        df = pd.read_csv(target, encoding="utf-8-sig", nrows=5)
        if "question" not in df.columns or "answer" not in df.columns:
            target.unlink()
            raise HTTPException(400, "CSV must contain 'question' and 'answer' columns")
        rows = sum(1 for _ in open(target, encoding="utf-8")) - 1
    except pd.errors.ParserError as e:
        target.unlink()
        raise HTTPException(400, f"Invalid CSV: {e}")

    return DatasetInfo(
        dataset_id=dataset_id,
        filename=file.filename,
        rows=max(rows, 0),
        size_bytes=target.stat().st_size,
    )


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str):
    p = DATASETS_DIR / f"{dataset_id}.csv"
    if not p.exists():
        raise HTTPException(404, "Dataset not found")
    p.unlink()
    return {"deleted": dataset_id}
