

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional
from fastapi import UploadFile

from app.service.transcription_pool import TranscriptionOutput,TranscriptionJob,worker_pool
from app.schemas.transcriptionmodel import TranscriptionRequest,TranscriptionResult,TranscriptionSegment,WordTimestamp,StreamingChunk
from app.service.audio_service import audio_service

class TranscriptionService:

    # ── Synchronous (full response) ───────────────────────────

    async def transcribe(
        self,
        upload: UploadFile,
        params: TranscriptionRequest,
        request_id: str,
        client_ip: Optional[str] = None,
    ) -> TranscriptionResult:
        
         wav_path: Optional[Path] = None
         try:
              
            # ── 1. Validate & convert audio ───────────────────
            raw_bytes = await upload.read()
            await upload.seek(0)  # rewind for re-read inside audio_service
            wav_path, duration = await audio_service.validate_and_prepare(
                upload, request_id
            )
            # ── 2. Transcribe ─────────────────────────────────
            job = TranscriptionJob(
                job_id=request_id,
                audio_path=wav_path,
                language=params.language,
                task=params.task,
                beam_size=params.beam_size,
                word_timestamps=params.word_timestamps,
                vad_filter=params.vad_filter,
                initial_prompt=params.initial_prompt,
            )
            output: TranscriptionOutput = await worker_pool.transcribe(job)

            # ── 3. Build response ─────────────────────────────
            result = self._build_result(output, request_id)
            return result
         
         finally:
            if wav_path:
                audio_service.cleanup(wav_path)

    # ── Streaming (SSE / chunked) ─────────────────────────────

    async def transcribe_stream(
        self,
        upload: UploadFile,
        params: TranscriptionRequest,
        request_id: str,
    ) -> AsyncIterator[StreamingChunk]:
        """
        Yields StreamingChunk objects as segments complete.
        Caller (endpoint) formats these as Server-Sent Events.
        """
        wav_path: Optional[Path] = None
        try:
            wav_path, _ = await audio_service.validate_and_prepare(
                upload, request_id
            )

            # For streaming we run the worker pool job the same way
            # but emit each segment as it materialises.
            # Note: Faster-Whisper's transcribe() returns a lazy iterator
            # so we can yield segments progressively.
            job = TranscriptionJob(
                job_id=request_id,
                audio_path=wav_path,
                language=params.language,
                task=params.task,
                beam_size=params.beam_size,
                word_timestamps=params.word_timestamps,
                vad_filter=params.vad_filter,
                initial_prompt=params.initial_prompt,
            )

            # Run the full job (non-streaming from pool perspective)
            # then stream segments to caller one by one
            output: TranscriptionOutput = await worker_pool.transcribe(job)

            for i, seg in enumerate(output.segments):
                is_final = (i == len(output.segments) - 1)
                words = [
                    WordTimestamp(
                        word=w.word, start=w.start, end=w.end,
                        probability=w.probability
                    )
                    for w in seg.words
                ] if seg.words else None

                yield StreamingChunk(
                    segment_id=seg.id,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    is_final=is_final,
                    words=words,
                )
        finally:
            if wav_path:
                audio_service.cleanup(wav_path)
        

    # ── Private helpers ───────────────────────────────────────

    @staticmethod
    def _build_result(
        output: TranscriptionOutput, job_id: str
    ) -> TranscriptionResult:
        segments = [
            TranscriptionSegment(
                id=s.id,
                seek=s.seek,
                start=s.start,
                end=s.end,
                text=s.text,
                tokens=s.tokens,
                temperature=s.temperature,
                avg_logprob=s.avg_logprob,
                compression_ratio=s.compression_ratio,
                no_speech_prob=s.no_speech_prob,
                words=[
                    WordTimestamp(
                        word=w.word, start=w.start,
                        end=w.end, probability=w.probability
                    )
                    for w in s.words
                ] if s.words else None,
            )
            for s in output.segments
        ]
        return TranscriptionResult(
            job_id=uuid.UUID(job_id) if len(job_id) == 36 else uuid.uuid4(),
            status="completed",
            text=output.text,
            language=output.language,
            language_probability=output.language_probability,
            duration_seconds=output.duration_seconds,
            processing_time_seconds=output.processing_time_seconds,
            segments=segments,
            model_size=output.model_size,
            created_at=datetime.now(timezone.utc),
        )

        


# Singleton
transcription_service = TranscriptionService()