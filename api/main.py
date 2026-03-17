# ==========================================
# FastAPI Backend — Image Tampering Detection
# ==========================================
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model_service import ModelService

# ── Globals ────────────────────────────────────
model_service: ModelService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model once at startup."""
    global model_service

    weights_path = os.environ.get(
        "MODEL_WEIGHTS_PATH",
        os.path.join(os.path.dirname(__file__), "..", "tampering_model.pth"),
    )
    weights_path = os.path.abspath(weights_path)

    if not os.path.exists(weights_path):
        raise RuntimeError(
            f"Model weights not found at {weights_path}. "
            "Set MODEL_WEIGHTS_PATH environment variable."
        )

    print(f"[Startup] Loading model from {weights_path} ...")
    model_service = ModelService(weights_path)
    print("[Startup] Model loaded successfully.")
    yield
    print("[Shutdown] Cleaning up.")


app = FastAPI(
    title="Image Tampering Detection API",
    description="Detects image tampering using ELA + ResNet50 + Grad-CAM",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow configurable origins ──────────
allowed_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────
@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": model_service is not None,
    }


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Upload an image file and receive a tampering analysis.

    Returns JSON with prediction, confidence, Grad-CAM metrics,
    and base64-encoded images (original, ELA, and Grad-CAM overlay).
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/tiff", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Accepted: {', '.join(allowed_types)}",
        )

    # Read & analyse
    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        results = model_service.analyze(image_bytes)
        return results

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )
