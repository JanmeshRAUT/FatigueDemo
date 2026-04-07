import os
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile

from inference import run_inference
from model import get_onnx_session


app = FastAPI(title="Fatigue Detection API", version="1.0.0")
session = get_onnx_session()


@app.get("/")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, Any]:
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = run_inference(session, image_bytes)
        return {
            "status": "success",
            "prediction": result,
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
