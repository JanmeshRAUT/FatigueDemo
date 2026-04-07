import os
import asyncio
import base64
import binascii
import logging
import threading
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    from inference import analyze_frame, decode_image_from_bytes
    from model import get_onnx_session
    IMPORT_INIT_ERROR = None
except Exception as exc:
    analyze_frame = None
    decode_image_from_bytes = None
    get_onnx_session = None
    IMPORT_INIT_ERROR = exc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fatigue Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (updated ONLY when face is detected)
session = None
state_lock = threading.Lock()

last_prediction = "unknown"
last_confidence = 0.0
last_ear = 0.0
last_mar = 0.0


def _update_state(prediction: str, confidence: float, ear: float, mar: float) -> None:
    """Update global state when face is detected."""
    global last_prediction, last_confidence, last_ear, last_mar
    with state_lock:
        last_prediction = prediction
        last_confidence = float(confidence)
        last_ear = float(ear)
        last_mar = float(mar)
    logger.info(f"State updated: prediction={prediction}, confidence={confidence:.3f}, ear={ear:.4f}, mar={mar:.4f}")


def _get_current_state() -> dict[str, Any]:
    """Get current cached state (thread-safe)."""
    with state_lock:
        return {
            "prediction": last_prediction,
            "confidence": last_confidence,
            "ear": last_ear,
            "mar": last_mar,
        }


@app.on_event("startup")
async def on_startup() -> None:
    global session
    logger.info("FastAPI startup initialized")
    if IMPORT_INIT_ERROR is not None:
        logger.exception("Module import failed during startup", exc_info=IMPORT_INIT_ERROR)
        return

    try:
        session = get_onnx_session()
        logger.info("Model session initialized successfully")
    except Exception:
        session = None
        logger.exception("Model session failed to initialize")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP Exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Request validation error on {request.url.path}")
    return JSONResponse(
        status_code=400,
        content={"status": "error", "message": "Invalid request payload"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server error")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"}
    )


@app.get("/")
async def health_check() -> dict[str, str]:
    if IMPORT_INIT_ERROR is not None:
        return {"status": "error"}
    return {"status": "ok" if session is not None else "error"}


@app.get("/api/combined_data")
async def combined_data() -> dict[str, Any]:
    """Return current cached state to frontend."""
    state = _get_current_state()
    
    logger.debug(
        f"combined_data response: prediction={state['prediction']}, confidence={state['confidence']:.3f}"
    )

    return {
        "sensor": {
            "hr": 72,
            "temperature": 36.5,
            "spo2": 98,
        },
        "perclos": {
            "ear": state["ear"],
            "mar": state["mar"],
            "status": state["prediction"],
        },
        "prediction": {
            "status": state["prediction"],
            "confidence": state["confidence"],
        },
    }


@app.get("/api/vehicle/combined_data")
async def vehicle_combined_data() -> dict[str, Any]:
    """Return current cached state for vehicle dashboard."""
    state = _get_current_state()
    
    logger.debug(
        f"vehicle_combined_data response: prediction={state['prediction']}, confidence={state['confidence']:.3f}"
    )

    return {
        "speed": 0,
        "steering_angle": 0,
        "lane_status": "stable",
        "prediction": {
            "status": state["prediction"],
            "confidence": state["confidence"],
        },
    }


async def _run_inference(image) -> tuple[str, float, float, float]:
    """Run inference on image and return prediction, confidence, ear, mar."""
    if session is None:
        raise ValueError("Model not initialized")

    try:
        analysis = await asyncio.to_thread(analyze_frame, session, image)
        
        prediction = analysis.get("prediction", "unknown")
        confidence = analysis.get("confidence", 0.0)
        ear = analysis.get("ear", 0.0)
        mar = analysis.get("mar", 0.0)
        
        return prediction, confidence, ear, mar
    except Exception as exc:
        logger.error(f"Inference failed: {exc}")
        raise


@app.websocket("/ws/detect")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time frame processing."""
    await websocket.accept()
    logger.info(f"WebSocket connected from {websocket.client}")

    try:
        while True:
            try:
                # Receive JSON frame
                payload = await websocket.receive_json()
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected from {websocket.client}")
                break
            except Exception as exc:
                logger.warning(f"Invalid WebSocket JSON: {exc}")
                await websocket.send_json({
                    "status": "error",
                    "message": "Invalid JSON payload"
                })
                continue

            # Extract image data
            image_data = payload.get("image_data")
            if not image_data:
                await websocket.send_json({
                    "status": "error",
                    "message": "Missing image_data field"
                })
                continue

            try:
                # Decode base64
                if isinstance(image_data, str):
                    # Handle data:image/jpeg;base64,... format
                    if "," in image_data:
                        _, base64_data = image_data.split(",", 1)
                    else:
                        base64_data = image_data
                    
                    try:
                        frame_bytes = base64.b64decode(base64_data, validate=True)
                    except (binascii.Error, ValueError) as exc:
                        raise ValueError("Invalid base64 image data") from exc
                else:
                    raise ValueError("image_data must be a string")

                # Decode image
                image = decode_image_from_bytes(frame_bytes)

                # Run inference
                try:
                    prediction, confidence, ear, mar = await _run_inference(image)
                    
                    # Update state ONLY if inference succeeded
                    _update_state(prediction, confidence, ear, mar)
                    
                    # Send response
                    await websocket.send_json({
                        "status": "success",
                        "prediction": prediction,
                        "confidence": round(confidence, 3),
                    })
                    
                except ValueError as exc:
                    # Face not detected or inference failed - DON'T update state
                    logger.debug(f"Inference error: {exc}")
                    # Return last known state instead of erroring out
                    state = _get_current_state()
                    await websocket.send_json({
                        "status": "success",
                        "prediction": state["prediction"],
                        "confidence": round(state["confidence"], 3),
                    })

            except Exception as exc:
                logger.exception(f"Frame processing error: {exc}")
                await websocket.send_json({
                    "status": "error",
                    "message": "Frame processing failed"
                })

    except Exception as exc:
        logger.exception(f"WebSocket error: {exc}")
    finally:
        logger.info(f"WebSocket connection closed")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
