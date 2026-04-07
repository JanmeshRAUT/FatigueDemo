import os
import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from inference import run_inference
from model import get_onnx_session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])

app = FastAPI(title="Fatigue Detection API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

session = get_onnx_session()


# Custom exception handler for consistent error responses
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP Exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"status": "error", "message": "Rate limit exceeded. Try again later."}
    )


@app.get("/")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/predict")
@limiter.limit("5/minute")  # Stricter limit for prediction endpoint
async def predict(request: Request, image: UploadFile = File(...)) -> dict[str, Any]:
    logger.info(f"Received prediction request: {image.filename}")
    
    # Security: Validate content type
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if image.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG images are allowed.")
    
    # Security: Limit file size (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    file_size = 0
    image_bytes = b""
    chunk_size = 1024 * 1024  # 1MB chunks
    while True:
        chunk = await image.read(chunk_size)
        if not chunk:
            break
        file_size += len(chunk)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
        image_bytes += chunk
    
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result, confidence = run_inference(session, image_bytes)

        # Determine model input type
        input_meta = session.get_inputs()[0]
        input_shape = input_meta.shape
        if len(input_shape) == 2 and input_shape[1] == 13:
            model_input_type = "feature"
        else:
            model_input_type = "image"

        logger.info(f"Prediction completed: {result}, confidence: {confidence:.3f}")
        return {
            "status": "success",
            "prediction": result,
            "confidence": round(confidence, 3),
        }
    except ValueError as exc:
        logger.warning(f"Validation error: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Inference failed: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error during inference.") from exc


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
