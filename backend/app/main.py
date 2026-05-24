from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import ALLOWED_ORIGINS, FRONTEND_DIST
from app.routes import datasets, infer, models, train

app = FastAPI(title="Fine-tuning CPU MVP", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(train.router)
app.include_router(models.router)
app.include_router(infer.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
