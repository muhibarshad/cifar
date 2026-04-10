from __future__ import annotations

import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

from cifar10_inference import CIFAR10ResNet


MODEL = CIFAR10ResNet()


def _parse_cors_origins(raw_value: str | None) -> list[str]:
    if not raw_value:
        return ["http://localhost:3000", "http://127.0.0.1:3000"]

    origins = [origin.strip() for origin in raw_value.split(",")]
    return [origin for origin in origins if origin]


app = FastAPI(title="CIFAR-10 Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(os.getenv("CORS_ORIGINS")),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionItem(BaseModel):
    label: str
    probability: float


class PredictionResponse(BaseModel):
    filename: str | None = None
    top_class: str
    top_probability: float
    predictions: list[PredictionItem]


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "CIFAR-10 Prediction API",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/classes")
def classes() -> dict[str, list[str]]:
    return {"classes": MODEL.class_names}


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)) -> PredictionResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        with Image.open(io.BytesIO(content)) as pil_image:
            probabilities = MODEL.predict_pil(pil_image)
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Unsupported image file.") from exc

    ranked_indices = list(probabilities.argsort()[::-1])
    predictions = [
        PredictionItem(
            label=MODEL.class_names[index],
            probability=float(probabilities[index]),
        )
        for index in ranked_indices
    ]
    top_prediction = predictions[0]

    return PredictionResponse(
        filename=file.filename,
        top_class=top_prediction.label,
        top_probability=top_prediction.probability,
        predictions=predictions,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
