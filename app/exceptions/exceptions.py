

from __future__ import annotations

from http import HTTPStatus
from typing import Any, Dict, Optional


class STTBaseException(Exception):
    """Root of all custom exceptions in this service."""

    http_status: int = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code: str = "INTERNAL_ERROR"
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message or self.__class__.message
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }

# ── Audio Errors ──────────────────────────────────────────────────────────────

class AudioValidationError(STTBaseException):
    http_status = HTTPStatus.UNPROCESSABLE_ENTITY
    error_code = "AUDIO_VALIDATION_ERROR"
    message = "The uploaded audio file failed validation."


class AudioFormatError(AudioValidationError):
    error_code = "AUDIO_FORMAT_ERROR"
    message = "Unsupported audio format."


class AudioTooLargeError(AudioValidationError):
    error_code = "AUDIO_TOO_LARGE"
    message = "Audio file exceeds the maximum allowed size."


class AudioTooLongError(AudioValidationError):
    error_code = "AUDIO_TOO_LONG"
    message = "Audio duration exceeds the maximum allowed length."


class AudioCorruptedError(AudioValidationError):
    error_code = "AUDIO_CORRUPTED"
    message = "The audio file appears to be corrupted or unreadable."


# ── Transcription Errors ──────────────────────────────────────────────────────

class TranscriptionError(STTBaseException):
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code = "TRANSCRIPTION_ERROR"
    message = "Transcription failed."


class ModelNotReadyError(TranscriptionError):
    http_status = HTTPStatus.SERVICE_UNAVAILABLE
    error_code = "MODEL_NOT_READY"
    message = "The speech model is not yet loaded. Retry shortly."


class TranscriptionTimeoutError(TranscriptionError):
    http_status = HTTPStatus.GATEWAY_TIMEOUT
    error_code = "TRANSCRIPTION_TIMEOUT"
    message = "Transcription job timed out."


class WorkerPoolExhaustedError(TranscriptionError):
    http_status = HTTPStatus.TOO_MANY_REQUESTS
    error_code = "WORKER_POOL_EXHAUSTED"
    message = "All transcription workers are busy. Retry shortly."


# ── Auth / Rate Limit Errors ──────────────────────────────────────────────────
class RateLimitExceededError(STTBaseException):
    http_status = HTTPStatus.TOO_MANY_REQUESTS
    error_code = "RATE_LIMIT_EXCEEDED"
    message = "Too many requests. Please slow down."

# ── Infrastructure Errors ─────────────────────────────────────────────────────

class CacheError(STTBaseException):
    error_code = "CACHE_ERROR"
    message = "Cache operation failed."


class DatabaseError(STTBaseException):
    error_code = "DATABASE_ERROR"
    message = "Database operation failed."
