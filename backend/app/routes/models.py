import shutil
from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.config import MODELS_DIR
from app.schemas import ModelInfo

router = APIRouter(prefix="/api/models", tags=["models"])


def _dir_size(path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


@router.get("", response_model=list[ModelInfo])
def list_models():
    out = []
    if not MODELS_DIR.exists():
        return out
    for p in sorted(MODELS_DIR.iterdir()):
        if not p.is_dir():
            continue
        if not (p / "adapter_config.json").exists():
            continue
        out.append(ModelInfo(
            model_id=p.name,
            path=str(p),
            created_at=datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            size_bytes=_dir_size(p),
        ))
    return out


@router.delete("/{model_id}")
def delete_model(model_id: str):
    p = MODELS_DIR / model_id
    if not p.exists() or not p.is_dir():
        raise HTTPException(404, "Model not found")
    shutil.rmtree(p)
    return {"deleted": model_id}
