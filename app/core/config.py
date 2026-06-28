

from typing import List

from pathlib import Path
from anyio.functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

from pydantic import AnyHttpUrl, field_validator, model_validator



class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


    # ── Application ──────────────────────────────────────────
    APP_NAME: str = "STT-Service"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Server ───────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    RELOAD: bool = False

    # ── Whisper ──────────────────────────────────────────────
    WHISPER_MODEL_SIZE: str = "medium"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_DOWNLOAD_ROOT: str = "./models"
    WHISPER_NUM_WORKERS: int = 2
    WHISPER_BEAM_SIZE: int = 5
    WHISPER_VAD_FILTER: bool = True
    WHISPER_WORD_TIMESTAMPS: bool = True

    # ── CORS ─────────────────────────────────────────────────
    CORS_ORIGINS: str = "http://localhost:4200,http://localhost:3000"
    CORS_ALLOW_CREDENTIALS: bool = True

    # ── Metrics ──────────────────────────────────────────────
    METRICS_ENABLED: bool = True
    METRICS_PATH: str = "/metrics"

    # ── Audio ────────────────────────────────────────────────
    MAX_AUDIO_SIZE_MB: int = 25
    MAX_AUDIO_DURATION_SECONDS: int = 300
    ALLOWED_AUDIO_FORMATS: str = "wav,mp3,ogg,webm,m4a,flac,mp4"
    TEMP_AUDIO_DIR: str = "/tmp/stt_audio"
    AUDIO_CLEANUP_INTERVAL_SECONDS: int = 300


    def is_production(self) -> bool:
        return self.APP_ENV == "production"
    
    # ── Derived helpers ──────────────────────────────────────
    @property
    def allowed_audio_formats_list(self) -> List[str]:
        return [f.strip().lower() for f in self.ALLOWED_AUDIO_FORMATS.split(",")]
    
    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]
    
    @model_validator(mode="after")
    def _ensure_temp_dir(self) -> "Settings":
        Path(self.TEMP_AUDIO_DIR).mkdir(parents=True, exist_ok=True)
        return self
    

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton — call this everywhere instead of instantiating Settings()."""
    return Settings()


# Module-level convenience alias
settings: Settings = get_settings()