import os
import asyncio
import base64
import binascii
import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    SLOWAPI_IMPORT_ERROR = None
except Exception as exc:
    Limiter = None
    _rate_limit_exceeded_handler = None
    get_remote_address = None
    RateLimitExceeded = None
    SlowAPIMiddleware = None
    SLOWAPI_IMPORT_ERROR = exc

try:
    from inference import run_inference, run_inference_from_image, decode_image_from_bytes
    from model import get_onnx_session
    IMPORT_INIT_ERROR = None
except Exception as exc:
    run_inference = None
    run_inference_from_image = None
    decode_image_from_bytes = None
    get_onnx_session = None
    IMPORT_INIT_ERROR = exc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if Limiter is not None:
    limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])
else:
    class _NoopLimiter:
        def limit(self, _rule: str):
            def _decorator(func):
                return func
            return _decorator

    limiter = _NoopLimiter()

app = FastAPI(title="Fatigue Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if Limiter is not None:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
else:
    logger.warning("slowapi is unavailable; rate limiting is disabled")

session = None


@app.on_event("startup")
async def on_startup() -> None:
    global session
    logger.info("FastAPI startup initialized")
    if SLOWAPI_IMPORT_ERROR is not None:
        logger.exception("slowapi import failed", exc_info=SLOWAPI_IMPORT_ERROR)
    if IMPORT_INIT_ERROR is not None:
        logger.exception("Module import failed during startup", exc_info=IMPORT_INIT_ERROR)
        return

    try:
        session = get_onnx_session()
        logger.info("Model session initialized successfully")
    except Exception:
        session = None
        logger.exception("Model session failed to initialize")


# Custom exception handler for consistent error responses
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP Exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Request validation error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=400,
        content={"status": "error", "message": "Invalid request payload. Send multipart/form-data with file key 'file'."},
    )


if RateLimitExceeded is not None:
    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={"status": "error", "message": "Rate limit exceeded. Try again later."}
        )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server error")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Unexpected server error."}
    )


@app.get("/")
async def health_check() -> dict[str, str]:
    if IMPORT_INIT_ERROR is not None:
        return {"status": "error"}
    return {"status": "ok" if session is not None else "error"}


def _validate_upload_metadata(content_type: str | None) -> None:
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG images are allowed.")


async def _read_upload_bytes(file: UploadFile) -> bytes:
    max_file_size = 10 * 1024 * 1024
    image_bytes = b""
    chunk_size = 1024 * 1024

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        image_bytes += chunk
        if len(image_bytes) > max_file_size:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return image_bytes


async def _predict_from_bytes(route_name: str, image_bytes: bytes) -> dict[str, Any]:
    if IMPORT_INIT_ERROR is not None:
        raise HTTPException(status_code=503, detail="Service initialization failed. Check startup logs.")

    if session is None:
        raise HTTPException(status_code=503, detail="Model is not available. Please try again later.")

    logger.info(f"Processing prediction from {route_name} ({len(image_bytes)} bytes)")

    try:
        result, confidence = await asyncio.to_thread(run_inference, session, image_bytes)
        logger.info(f"Prediction completed for {route_name}: {result}, confidence: {confidence:.3f}")
        return {
            "status": "success",
            "prediction": result,
            "confidence": round(confidence, 3),
        }
    except ValueError as exc:
        logger.warning(f"Validation error for {route_name}: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception(f"Inference failed for {route_name}")
        raise HTTPException(status_code=500, detail="Internal server error during inference.")


async def _predict_from_image(route_name: str, image) -> dict[str, Any]:
    if IMPORT_INIT_ERROR is not None:
        raise HTTPException(status_code=503, detail="Service initialization failed. Check startup logs.")

    if session is None:
        raise HTTPException(status_code=503, detail="Model is not available. Please try again later.")

    logger.info(f"Processing prediction from {route_name} (decoded image)")

    try:
        result, confidence = await asyncio.to_thread(run_inference_from_image, session, image)
        logger.info(f"Prediction completed for {route_name}: {result}, confidence: {confidence:.3f}")
        return {
            "status": "success",
            "prediction": result,
            "confidence": round(confidence, 3),
        }
    except ValueError as exc:
        logger.warning(f"Validation error for {route_name}: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception(f"Inference failed for {route_name}")
        raise HTTPException(status_code=500, detail="Internal server error during inference.")


@app.post("/api/v1/predict")
@app.post("/predict")
@limiter.limit("5/minute")  # Stricter limit for prediction endpoint
async def predict(request: Request, file: UploadFile = File(..., alias="file")) -> dict[str, Any]:
    logger.info(f"Received HTTP prediction request on {request.url.path}: {file.filename}")
    _validate_upload_metadata(file.content_type)
    image_bytes = await _read_upload_bytes(file)
    return await _predict_from_bytes(f"http:{request.url.path}", image_bytes)


@app.websocket("/ws")
@app.websocket("/ws/detect")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"WebSocket connection opened on {websocket.url.path} from {websocket.client}")

    try:
        while True:
            try:
                payload = await websocket.receive_json()
            except WebSocketDisconnect:
                logger.info(f"WebSocket connection closed by client: {websocket.client}")
                break
            except Exception as exc:
                logger.warning(f"Invalid WebSocket JSON from {websocket.client}: {exc}")
                await websocket.send_json({"status": "error", "message": "Invalid JSON payload. Expected {\"image_data\": \"data:image/jpeg;base64,...\"}."})
                continue

            logger.info(f"WebSocket frame received on {websocket.url.path} from {websocket.client}")
            image_data = payload.get("image_data") if isinstance(payload, dict) else None
            if not image_data:
                await websocket.send_json({"status": "error", "message": "Missing image_data field."})
                continue

            try:
                if not isinstance(image_data, str):
                    raise ValueError("image_data must be a base64 string.")

                if "," in image_data:
                    _, base64_data = image_data.split(",", 1)
                else:
                    base64_data = image_data

                try:
                    frame_bytes = base64.b64decode(base64_data, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise ValueError("Invalid base64 image data.") from exc

                image = decode_image_from_bytes(frame_bytes)
                response = await _predict_from_image("ws:/ws", image)
                await websocket.send_json(response)
            except HTTPException as exc:
                await websocket.send_json({"status": "error", "message": exc.detail})
            except Exception as exc:
                logger.exception("Unexpected WebSocket processing error")
                await websocket.send_json({"status": "error", "message": str(exc) or "Internal server error during streaming inference."})
    except Exception:
        logger.exception(f"WebSocket loop crashed unexpectedly for {websocket.client}")
    finally:
        logger.info(f"WebSocket connection closed on {websocket.url.path} for {websocket.client}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
