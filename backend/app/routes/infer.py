import os
import sys
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import BACKEND_ROOT, MODELS_DIR
from app.schemas import InferRequest, InferResponse

# Allow importing the existing modules in backend/ (one level above app/)
sys.path.insert(0, str(BACKEND_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

router = APIRouter(prefix="/api/infer", tags=["infer"])

# Lazy-loaded inference manager (cached by model_id)
_cache: dict = {}


def _load(model_id: str):
    if model_id in _cache:
        return _cache[model_id]
    model_dir = MODELS_DIR / model_id
    if not model_dir.exists():
        raise HTTPException(404, "Model not found")

    from model_manager_cpu import create_inference_manager as create_loader
    from inference_manager import create_inference_manager as create_inf

    loader = create_loader(str(model_dir))
    model, tokenizer = loader.load_finetuned_model()
    inf = create_inf(model, tokenizer)
    _cache[model_id] = inf
    return inf


@router.post("", response_model=InferResponse)
def infer(req: InferRequest):
    inf = _load(req.model_id)
    inf.update_generation_config(max_new_tokens=req.max_new_tokens, temperature=req.temperature)
    t0 = time.perf_counter()
    answer = inf.generate_response(req.question, max_new_tokens=req.max_new_tokens, temperature=req.temperature)
    return InferResponse(
        question=req.question,
        answer=answer,
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
    )
