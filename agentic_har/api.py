"""FastAPI entry point for the shared recognition pipeline."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .runtime import build_live_pipeline
from .schemas import RecognitionRequest, RecognitionResponse

logger = logging.getLogger(__name__)


def create_app(pipeline_factory=build_live_pipeline) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline_factory()
        yield

    app = FastAPI(title="Agentic HAR", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/recognize", response_model=RecognitionResponse)
    def recognize(request: RecognitionRequest):
        try:
            return app.state.pipeline.recognize(request)
        except Exception as exc:
            logger.error("Recognition failed (%s)", type(exc).__name__)
            raise HTTPException(status_code=502, detail="Recognition failed") from exc

    return app


app = create_app()
