

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.schemas.transcriptionmodel import TranscriptionRequest,TranscriptionResult
from app.service.transcription_service import transcription_service

router = APIRouter(prefix="/transcribe", tags=["Transcription"])

def _get_params(
    language: Optional[str] = Form(default=None),
    task: str = Form(default="transcribe"),
    beam_size: int = Form(default=5, ge=1, le=10),
    word_timestamps: bool = Form(default=True),
    vad_filter: bool = Form(default=True),
    initial_prompt: Optional[str] = Form(default=None),
) -> TranscriptionRequest:
    return TranscriptionRequest(
        language=language,
        task=task,
        beam_size=beam_size,
        word_timestamps=word_timestamps,
        vad_filter=vad_filter,
        initial_prompt=initial_prompt,
    )



@router.post(
    "",
    response_model=TranscriptionResult,
    summary="Transcribe audio file",
    description=(
        "Upload an audio file and receive a full transcription synchronously. "
        "Supports wav, mp3, ogg, webm, m4a, flac, mp4. "
        "Results are cached by audio content + parameters."
    ),
    responses={
        200: {"description": "Transcription completed successfully"},
        413: {"description": "Audio file too large"},
        422: {"description": "Invalid audio file or parameters"},
        429: {"description": "Rate limit exceeded or worker pool exhausted"},
        500: {"description": "Transcription failed"},
        503: {"description": "Model not yet loaded"},
    },
)
async def transcribe(
    request: Request,
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    params: TranscriptionRequest = Depends(_get_params),
)->TranscriptionResult:
     request_id: str = request.state.request_id
     client_ip: str = request.client.host if request.client else "unknown"
     return await transcription_service.transcribe(
        upload=audio,
        params=params,
        request_id=request_id,
        client_ip=client_ip,
    )

# ── POST /api/v1/transcribe/stream ────────────────────────────────────────────

@router.post(
    "/stream",
    summary="Transcribe audio (Server-Sent Events)",
    description=(
        "Streams transcription segments as they are produced. "
        "Response is `text/event-stream` (SSE). "
        "Each event contains a JSON-encoded `StreamingChunk`."
    ),
    response_class=StreamingResponse,
    responses={
        200: {"description": "SSE stream of transcription chunks"},
        401: {"description": "Invalid or missing API key"},
        422: {"description": "Invalid audio file or parameters"},
    },
)
async def transcribe_stream(
    request: Request,
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    params: TranscriptionRequest = Depends(_get_params),
) -> StreamingResponse:

    async def event_generator():
        request_id: str = request.state.request_id
        async for chunk in transcription_service.transcribe_stream(
            upload=audio,
            params=params,
            request_id=request_id,
        ):
            yield (
                f"data: {chunk.model_dump_json()}\n\n"
            )

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )

     