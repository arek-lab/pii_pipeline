from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from . import pipeline
from .errors import InternalMaskingError
from .models import DetectRequest, DetectResponse
from .singletons import get_morfeusz, get_ner_model, init_all


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_all()
    yield
    # No cleanup needed for current resources


app = FastAPI(
    title="pii-pipeline",
    description="PII detection and masking for Polish plain text.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/detect", response_model=DetectResponse)
async def detect_endpoint(request: DetectRequest) -> DetectResponse:
    try:
        return await pipeline.detect(request)
    except InternalMaskingError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health() -> dict:
    # TODO: re-enable model checks once singletons.init_all loads real models
    # if get_ner_model() is None or get_morfeusz() is None:
    #     raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ok"}
