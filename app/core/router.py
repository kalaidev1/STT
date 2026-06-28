from fastapi import APIRouter

from app.api.transcription import router as transcription_router


api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(transcription_router)