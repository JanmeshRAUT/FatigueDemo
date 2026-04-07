import os
import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile

from inference import run_inference
from model import get_onnx_session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fatigue Detection API", version="1.0.0")
session = get_onnx_session()


@app.get("/")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, Any]:
    logger.info(f"Received prediction request: {image.filename}")
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

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
            "model_input_type": model_input_type,
        }
    except ValueError as exc:
        logger.warning(f"Validation error: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Inference failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
