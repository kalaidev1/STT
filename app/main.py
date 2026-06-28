from fastapi import FastAPI

import asyncio
from fastapi.middleware.cors import CORSMiddleware
from typer.cli import app
from app.core.logging import get_logger,configure_logging
from app.core.config import settings
from app.utils.request_id import RequestIDMiddleware
from app.exceptions.error_handlear import (
    stt_exception_handler,
    unhandled_exception_handler,
)
from app.exceptions.exceptions import STTBaseException

from app.core.router import api_v1_router
from prometheus_fastapi_instrumentator import Instrumentator
from app.service.transcription_pool import worker_pool

from app.service.audio_service import AudioService


configure_logging()
logger = get_logger(__name__)

async def lifespan(app: FastAPI):
    # ── STARTUP ──────────────────────────────────────────────
    logger.info("app.starting", env=settings.APP_ENV, version=settings.APP_VERSION)
    await worker_pool.start()
        # Background task: clean up stale temp audio files every N seconds
    async def _periodic_cleanup():
        while True:
            await asyncio.sleep(settings.AUDIO_CLEANUP_INTERVAL_SECONDS)
            await AudioService.cleanup_old_files(
                max_age_seconds=settings.AUDIO_CLEANUP_INTERVAL_SECONDS
            )

    cleanup_task = asyncio.create_task(_periodic_cleanup())
    logger.info("app.ready")
    yield  # ← application is running

    # ── SHUTDOWN ─────────────────────────────────────────────
    logger.info("app.shutting_down")
    cleanup_task.cancel()
    await worker_pool.stop()
    logger.info("app.stopped")





def create_app() -> FastAPI:
     app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Enterprise-grade Speech-to-Text service powered by Faster-Whisper. "
            "Supports multiple audio formats, streaming, multi-language detection, "
            "word-level timestamps, and result caching."
        ),
        openapi_url="/api/v1/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
     

    # ── Middleware (outermost first) ────────────────────────── 
     app.add_middleware(RequestIDMiddleware)
     app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
        )
     
    # ── Exception handlers ────────────────────────────────────
     app.add_exception_handler(STTBaseException, stt_exception_handler)
     app.add_exception_handler(Exception, unhandled_exception_handler)

      # ── Prometheus ────────────────────────────────────────────
     if settings.METRICS_ENABLED:
        Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            excluded_handlers=["/health", "/ready", settings.METRICS_PATH],
        ).instrument(app).expose(app, endpoint=settings.METRICS_PATH)

    # ── Routes ────────────────────────────────────────────────
     app.include_router(api_v1_router)

     return app


app = create_app()
