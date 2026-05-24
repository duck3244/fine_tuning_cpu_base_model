from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
STORAGE_ROOT = BACKEND_ROOT / "storage"
DATASETS_DIR = STORAGE_ROOT / "datasets"
MODELS_DIR = STORAGE_ROOT / "models"
LOGS_DIR = STORAGE_ROOT / "logs"

for d in (DATASETS_DIR, MODELS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

FRONTEND_DIST = BACKEND_ROOT.parent / "frontend" / "dist"
ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
