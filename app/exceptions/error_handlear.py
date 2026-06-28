from __future__ import annotations

import traceback
from http import HTTPStatus

from fastapi import Request
from fastapi.responses import JSONResponse

from app.exceptions.exceptions import STTBaseException
from app.core.logging import get_logger

logger = get_logger(__name__)



async def stt_exception_handler(
    request: Request, exc: STTBaseException
) -> JSONResponse:
    """Handler for all domain-specific exceptions."""
    request_id = request.headers.get("X-Request-ID", "unknown")

    logger.warning(
        "request.domain_error",
        request_id=request_id,
        error_code=exc.error_code,
        message=exc.message,
        path=str(request.url.path),
    )

    payload = exc.to_dict()
    payload["request_id"] = request_id

    return JSONResponse(
        status_code=exc.http_status,
        content=payload,
    )


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all for unexpected exceptions — never leak stack traces."""
    request_id = request.headers.get("X-Request-ID", "unknown")

    logger.error(
        "request.unhandled_error",
        request_id=request_id,
        path=str(request.url.path),
        error=str(exc),
        traceback=traceback.format_exc(),
    )

    return JSONResponse(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        content={
            "error_code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred. Please try again later.",
            "request_id": request_id,
            "details": {},
        },
    )
