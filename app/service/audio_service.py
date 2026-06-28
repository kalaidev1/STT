
from __future__ import annotations

import asyncio
import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

import ffmpeg
import soundfile as sf
from fastapi import UploadFile

from app.core.config import settings
from app.exceptions.exceptions import AudioTooLargeError,AudioFormatError,AudioTooLongError,AudioCorruptedError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Whisper works best on 16 kHz mono PCM WAV
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1

class AudioService:
    async def validate_and_prepare(
        self, upload: UploadFile, request_id: str
    ) -> Tuple[Path, float]:
        
         raw_bytes = await upload.read()
         # 1 — Size check (fast, before any disk I/O)
         self._check_size(len(raw_bytes), upload.filename)

         # 2 — Format check
         extension = self._extract_extension(upload.filename, upload.content_type)
         self._check_format(extension)

         # 3 — Write raw upload to temp dir
         raw_path = self._temp_path(f"raw_{request_id}", extension)
         raw_path.write_bytes(raw_bytes)

         # 4 — Convert to WAV + probe duration
         wav_path, duration = await self._convert_to_wav(raw_path, request_id)

         # 5 — Duration check
         self._check_duration(duration)

         # 6 — Record metrics

         logger.info(
            "audio.prepared",
            request_id=request_id,
            filename=upload.filename,
            size_bytes=len(raw_bytes),
            duration=round(duration, 2),
            format=extension,
         )

         # Clean up the raw upload; keep only the converted WAV
         raw_path.unlink(missing_ok=True)

         return wav_path, duration

    def cleanup(self, path: Path) -> None:
            """Remove a temp file after transcription is complete."""
            try:
                path.unlink(missing_ok=True)
                logger.debug("audio.cleanup", path=str(path))
            except Exception as exc:
                logger.warning("audio.cleanup_failed", path=str(path), error=str(exc))

    
    # ── Validation helpers ────────────────────────────────────

    def _check_size(self, size_bytes: int, filename: Optional[str]) -> None:
        max_bytes = settings.MAX_AUDIO_SIZE_MB * 1024 * 1024
        if size_bytes > max_bytes:
            raise AudioTooLargeError(
                f"File '{filename}' is {size_bytes / 1_048_576:.1f} MB. "
                f"Maximum allowed: {settings.MAX_AUDIO_SIZE_MB} MB.",
                details={"size_mb": round(size_bytes / 1_048_576, 2),
                         "limit_mb": settings.MAX_AUDIO_SIZE_MB},
            )
        
    def _extract_extension(
        self, filename: Optional[str], content_type: Optional[str]
    ) -> str:
        if filename:
            suffix = Path(filename).suffix.lstrip(".").lower()
            if suffix:
                return suffix

        # Fallback: infer from MIME type
        mime_map = {
            "audio/wav": "wav",
            "audio/wave": "wav",
            "audio/mpeg": "mp3",
            "audio/mp3": "mp3",
            "audio/ogg": "ogg",
            "audio/webm": "webm",
            "audio/mp4": "m4a",
            "audio/x-m4a": "m4a",
            "audio/flac": "flac",
            "audio/x-flac": "flac",
            "video/webm": "webm",
            "video/mp4": "mp4",
        }
        ext = mime_map.get(content_type or "", "")
        if not ext:
            raise AudioFormatError(
                "Cannot determine audio format from filename or content-type.",
                details={"content_type": content_type},
            )
        return ext
    
    def _check_format(self, extension: str) -> None:
        if extension not in settings.allowed_audio_formats_list:
            raise AudioFormatError(
                f"Format '{extension}' is not supported.",
                details={
                    "provided": extension,
                    "allowed": settings.allowed_audio_formats_list,
                },
            )
        
    def _check_duration(self, duration: float) -> None:
        limit = settings.MAX_AUDIO_DURATION_SECONDS
        if duration > limit:
            raise AudioTooLongError(
                f"Audio is {duration:.0f}s. Maximum allowed: {limit}s.",
                details={"duration_seconds": round(duration, 2),
                         "limit_seconds": limit},
            )
    


    # ── FFmpeg conversion ─────────────────────────────────────

    async def _convert_to_wav(
        self, raw_path: Path, request_id: str
    ) -> Tuple[Path, float]:
        """
        Convert any audio format → 16 kHz mono PCM WAV using ffmpeg.
        Runs in a thread executor so it doesn't block the event loop.
        """
        wav_path = self._temp_path(f"wav_{request_id}", "wav")
        loop = asyncio.get_event_loop()
        try:
            duration = await loop.run_in_executor(
                None,
                self._ffmpeg_convert,
                raw_path,
                wav_path,
            )
        except ffmpeg.Error as exc:
            wav_path.unlink(missing_ok=True)
            raise AudioCorruptedError(
                "Could not decode audio. The file may be corrupted.",
                details={"ffmpeg_stderr": exc.stderr.decode(errors="replace")[-500:]},
            ) from exc
        return wav_path, duration

    def _ffmpeg_convert(self, src: Path, dst: Path) -> float:
        """Blocking ffmpeg call — always run in a thread."""
        try:
            (
                ffmpeg
                .input(str(src))
                .output(
                    str(dst),
                    ar=TARGET_SAMPLE_RATE,     # resample to 16 kHz
                    ac=TARGET_CHANNELS,         # mono
                    acodec="pcm_s16le",         # 16-bit PCM
                    f="wav",
                )
                .overwrite_output()
                .run(quiet=True)
            )
            # Probe the output to get an accurate duration
            probe = ffmpeg.probe(str(dst))
            duration = float(probe["format"]["duration"])
            return duration
        except (KeyError, ValueError):
            # Duration unavailable — estimate from file size
            info = sf.info(str(dst))
            return info.duration

    # ── Temp file management ──────────────────────────────────

    @staticmethod
    def _temp_path(stem: str, extension: str) -> Path:
        base = Path(settings.TEMP_AUDIO_DIR)
        return base / f"{stem}.{extension}"

    @staticmethod
    async def cleanup_old_files(max_age_seconds: int = 600) -> None:
        """
        Periodic cleanup task — remove temp files older than max_age_seconds.
        Schedule with asyncio in main.py.
        """
        temp_dir = Path(settings.TEMP_AUDIO_DIR)
        now = time.time()
        removed = 0
        for f in temp_dir.glob("*.wav"):
            try:
                if now - f.stat().st_mtime > max_age_seconds:
                    f.unlink()
                    removed += 1
            except Exception:
                pass
        if removed:
            logger.info("audio.cleanup_sweep", removed=removed)

# Singleton
audio_service = AudioService()