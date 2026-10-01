"""Local-only serving; HTTP timing stays separate from direct model timings."""
import asyncio
import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal
from uuid import uuid4
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from starlette.concurrency import run_in_threadpool
from quantbench.common import SUPPORTED_VARIANTS
from quantbench.runners.local import LocalRunner

Text = Annotated[str, StringConstraints(min_length=1, max_length=4096, strict=True)]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    texts: list[Text] = Field(min_length=1, max_length=8)
    variant: Literal["pytorch_fp32", "onnx_fp32", "onnx_int8", "onnx_static_int8"]


class Prediction(BaseModel):
    label: Literal["NEGATIVE", "POSITIVE"]
    scores: list[float]


class PredictResponse(BaseModel):
    request_id: str
    variant: str
    model_revision: str
    artifact_hash: str
    predictions: list[Prediction]


def create_app(variant="onnx_int8", runner_factory=LocalRunner):
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError("Unsupported serving variant")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    @asynccontextmanager
    async def lifespan(app):
        runner = await run_in_threadpool(runner_factory, variant)
        await run_in_threadpool(runner.load)
        app.state.runner = runner
        app.state.inference_lock = asyncio.Lock()
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            app.state.runner = None

    app = FastAPI(title="Local sentiment inference", lifespan=lifespan)
    app.state.ready = False

    @app.get("/health/ready")
    async def ready():
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="Model not ready")
        return {"ready": True, "variant": variant}

    @app.post("/v1/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest):
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="Model not ready")
        if request.variant != variant:
            raise HTTPException(status_code=400, detail="Variant not loaded by this server")
        if any(not text.strip() for text in request.texts):
            raise HTTPException(status_code=422, detail="Blank text is not allowed")
        if app.state.inference_lock.locked():
            raise HTTPException(status_code=503, detail="Server busy; retry later")
        runner = app.state.runner

        def infer():
            inputs = runner.prepare(runner.preprocess(request.texts, 256, pad_to_max=False))
            return runner.predict(inputs)

        async with app.state.inference_lock:
            logits = await run_in_threadpool(infer)
        if logits.shape != (len(request.texts), 2) or not np.isfinite(logits).all():
            raise HTTPException(status_code=500, detail="Invalid model output")
        scores = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        scores /= scores.sum(axis=1, keepdims=True)
        manifest = runner.manifest
        weight = manifest.get("graph", "model/model.safetensors")
        return PredictResponse(request_id=str(uuid4()), variant=variant,
                               model_revision=manifest["model_revision"], artifact_hash=manifest["files"][weight],
                               predictions=[Prediction(label=("NEGATIVE", "POSITIVE")[int(row.argmax())],
                                                       scores=row.tolist()) for row in scores])

    return app
