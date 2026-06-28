from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, Field, field_validator

class TranscriptionRequest(BaseModel):
    """
    Query parameters accepted alongside the uploaded audio file.
    All fields are optional — sensible defaults come from settings.
    """

    language: Optional[str] = Field(
        default=None,
        description=(
            "BCP-47 language code of the audio (e.g. 'en', 'ta', 'hi'). "
            "Leave null to trigger automatic language detection."
        ),
        examples=["en", "ta", "hi", None],
    )
    task: str = Field(
        default="transcribe",
        description="'transcribe' returns text in the source language; "
                    "'translate' returns English regardless of input language.",
        pattern="^(transcribe|translate)$",
    )
    beam_size: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Beam search width. Higher = more accurate but slower.",
    )
    word_timestamps: bool = Field(
        default=True,
        description="Return per-word start/end timestamps.",
    )
    vad_filter: bool = Field(
        default=True,
        description="Skip silent segments using Voice Activity Detection.",
    )
    initial_prompt: Optional[str] = Field(
        default=None,
        max_length=512,
        description="Optional text to prime the model context (e.g. domain jargon).",
    )

    @field_validator("language")
    @classmethod
    def _normalise_language(cls, v: Optional[str]) -> Optional[str]:
        return v.strip().lower() if v else None
    
# ─────────────────────────────────────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class WordTimestamp(BaseModel):
    word: str
    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    probability: float = Field(description="Model confidence 0-1")


class TranscriptionSegment(BaseModel):
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
    words: Optional[List[WordTimestamp]] = None


class TranscriptionResult(BaseModel):
    """Full response body returned by POST /api/v1/transcribe"""

    job_id: UUID = Field(description="Unique identifier for this transcription job")
    status: str = Field(description="always 'completed' for sync endpoint")
    text: str = Field(description="Full transcription text")
    language: str = Field(description="Detected or requested language code")
    language_probability: float = Field(
        description="Confidence of detected language (0-1)"
    )
    duration_seconds: float = Field(description="Audio duration processed")
    processing_time_seconds: float = Field(
        description="Wall-clock time taken for transcription"
    )
    segments: List[TranscriptionSegment] = Field(
        default_factory=list,
        description="Per-segment breakdown with timestamps",
    )
    model_size: str = Field(description="Whisper model variant used")
    created_at: datetime = Field(description="UTC timestamp of completion")

    model_config = {"from_attributes": True}


class StreamingChunk(BaseModel):
    """One SSE payload item for streaming endpoint."""

    segment_id: int
    start: float
    end: float
    text: str
    is_final: bool = False
    words: Optional[List[WordTimestamp]] = None


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: dict = Field(default_factory=dict)
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str                        # ok | degraded | down
    model_loaded: bool
    worker_pool_active: int
    worker_pool_capacity: int
    uptime_seconds: float
    version: str


class ReadinessResponse(BaseModel):
    ready: bool
    checks: dict
