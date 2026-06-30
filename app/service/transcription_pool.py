

from __future__ import annotations

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo

from app.exceptions.exceptions import TranscriptionTimeoutError,TranscriptionError,ModelNotReadyError
from app.core.config import settings
from app.core.logging import get_logger


logger = get_logger(__name__)

# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class TranscriptionJob:
    job_id: str
    audio_path: Path
    language: Optional[str] = None
    task: str = "transcribe"
    beam_size: int = 5
    word_timestamps: bool = True
    vad_filter: bool = True
    initial_prompt: Optional[str] = None


@dataclass
class WordResult:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class SegmentResult:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: List[WordResult] = field(default_factory=list)


@dataclass
class TranscriptionOutput:
    job_id: str
    text: str
    language: str
    language_probability: float
    duration_seconds: float
    processing_time_seconds: float
    model_size: str



class TranscriptionWorkerPool:
     
     def __init__(self) -> None:
        self._model: Optional[WhisperModel] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._ready: bool = False
        self._queued: int = 0


     # ── Lifecycle ─────────────────────────────────────────────

     async def start(self) -> None:
        """Load the model and prepare the pool. Called once at app startup."""
        n = settings.WHISPER_NUM_WORKERS
        logger.info(
            "worker_pool.loading",
            model=settings.WHISPER_MODEL_SIZE,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            workers=n,
        )
        loop = asyncio.get_event_loop()
        # Load model in executor so startup doesn't block the event loop
        self._model = await loop.run_in_executor(
            None, self._load_model
        )
        self._semaphore = asyncio.Semaphore(n)
        self._executor = ThreadPoolExecutor(
            max_workers=n,
            thread_name_prefix="whisper-worker",
        )
        self._ready = True
        logger.info("worker_pool.ready", workers=n)

     
     async def stop(self) -> None:
        """Graceful shutdown — wait for in-flight jobs to finish."""
        self._ready = False
        if self._executor:
            self._executor.shutdown(wait=True)
        logger.info("worker_pool.stopped")

     def _load_model(self) -> WhisperModel:
        return WhisperModel(
            settings.WHISPER_MODEL_SIZE,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            download_root=settings.WHISPER_DOWNLOAD_ROOT,
            num_workers=settings.WHISPER_NUM_WORKERS,
        )

         # ── Public API ────────────────────────────────────────────

     @property
     def is_ready(self) -> bool:
        return self._ready

     @property
     def active_workers(self) -> int:
        if self._semaphore is None:
            return 0
        return settings.WHISPER_NUM_WORKERS - self._semaphore._value

     @property
     def queued_jobs(self) -> int:
        return self._queued



     async def transcribe(
        self,
        job: TranscriptionJob,
        timeout: float = 120.0,
     ) -> TranscriptionOutput:
        """
        Submit a transcription job.
        Blocks until a worker slot is free, then runs transcription in a thread.
        """
        if not self._ready:
            raise ModelNotReadyError()

        # If semaphore has no free slots at all, reject immediately
        if self._semaphore._value == 0:
            self._queued += 1

        t_submit = time.monotonic()
        try:
            async with asyncio.timeout(timeout):
                async with self._semaphore:
                    self._queued = max(0, self._queued - 1)
            
                    result = await self._run_in_thread(job)
                    return result
                
        except TimeoutError as exc:
           
            raise TranscriptionTimeoutError(
                f"Job {job.job_id} timed out after {timeout}s"
            ) from exc

     async def _run_in_thread(self, job: TranscriptionJob) -> TranscriptionOutput:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._transcribe_sync,
            job,
        )

    # ── Synchronous transcription (runs in thread) ────────────

     def _transcribe_sync(self, job: TranscriptionJob) -> TranscriptionOutput:
        if self._model is None:
            raise ModelNotReadyError()

        t0 = time.monotonic()
        lang_label = job.language or "auto"

        try:
            logger.info(
                "transcription.start",
                job_id=job.job_id,
                language=lang_label,
                audio_path=str(job.audio_path),
            )

            segments_iter: Iterator[Segment]
            info: TranscriptionInfo
            segments_iter, info = self._model.transcribe(
                str(job.audio_path),
                language=job.language,
                task=job.task,
                beam_size=job.beam_size,
                word_timestamps=job.word_timestamps,
                vad_filter=job.vad_filter,
                initial_prompt=job.initial_prompt,
            )

            # Materialise the lazy iterator
            raw_segments: List[Segment] = list(segments_iter)

            # ── Build output ──────────────────────────────────
            full_text = " ".join(s.text.strip() for s in raw_segments)
            segment_results: List[SegmentResult] = []

           

            elapsed = time.monotonic() - t0

            logger.info(
                "transcription.complete",
                job_id=job.job_id,
                language=info.language,
                language_prob=round(info.language_probability, 3),
                audio_duration=round(info.duration, 2),
                processing_time=round(elapsed, 2),
            )

            return TranscriptionOutput(
                job_id=job.job_id,
                text=full_text,
                language=info.language,
                language_probability=info.language_probability,
                duration_seconds=info.duration,
                processing_time_seconds=elapsed,
                model_size=settings.WHISPER_MODEL_SIZE,
            )

        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.error(
                "transcription.failed",
                job_id=job.job_id,
                error=str(exc),
                processing_time=round(elapsed, 2),
                exc_info=True,
            )
            raise TranscriptionError(
                f"Transcription failed: {exc}"
            ) from exc


# ── Singleton accessed by the application ─────────────────────────────────────
worker_pool = TranscriptionWorkerPool()
