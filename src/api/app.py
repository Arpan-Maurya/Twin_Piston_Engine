"""
FastAPI Application Factory — REST & Real-Time Server Entrypoint (Module 20).

Configures:
- Versioned API router namespace (/api/v1)
- Configurable CORS middleware
- Centralized exception handling (clean JSON error envelopes, zero stack traces/secret leakage)
- OpenAPI interactive documentation (/docs, /redoc)
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.v1 import api_v1_router
from src.api.v1.schemas import APIErrorResponse
from src.core.config import AppSettings, get_settings
from src.core.exceptions import (
    BoundaryViolationError,
    DigitalTwinError,
    InvalidChannelError,
    PacketIntegrityError,
    PistonEngineError,
    ResourceNotFoundException,
    ValidationException,
)
from src.core.logging import get_logger

logger = get_logger(__name__)


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Factory creating and configuring the FastAPI application instance."""
    app_settings = settings or get_settings()

    app = FastAPI(
        title="Aero Piston Engine Digital Twin REST & Real-Time API",
        description=(
            "Production-ready prototype backend API exposing real-time Digital Twin telemetry, "
            "thermodynamic/mechanical diagnostics, ML fault classification, Health Index, RUL estimation, "
            "mission risk assessment, decision-support advisories, and scenario replay/what-if simulation."
        ),
        version=app_settings.version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Configurable CORS Middleware
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # SEC-004: Request Payload Size Limit Middleware (1MB limit)
    @app.middleware("http")
    async def limit_request_size(request: Request, call_next):
        max_bytes = 1_048_576  # 1 MB maximum payload limit
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > max_bytes:
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content=APIErrorResponse(
                            error_code="PAYLOAD_TOO_LARGE",
                            message="Request payload exceeds maximum allowed size of 1MB.",
                        ).model_dump(mode="json"),
                    )
            except ValueError:
                pass
        return await call_next(request)

    # -------------------------------------------------------------------------
    # Exception Handlers
    # -------------------------------------------------------------------------

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        err_res = APIErrorResponse(
            error_code="HTTP_ERROR",
            message=exc.detail if isinstance(exc.detail, str) else "HTTP Exception",
            details={"status_code": exc.status_code},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=err_res.model_dump(mode="json"),
        )

    @app.exception_handler(PistonEngineError)
    async def domain_exception_handler(request: Request, exc: PistonEngineError) -> JSONResponse:
        logger.error(f"Domain exception on {request.url.path}: {exc}")
        status_code = status.HTTP_400_BAD_REQUEST
        if isinstance(exc, ResourceNotFoundException):
            status_code = status.HTTP_404_NOT_FOUND
        elif isinstance(exc, ValidationException):
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        elif isinstance(exc, DigitalTwinError):
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

        err_res = APIErrorResponse(
            error_code=exc.__class__.__name__,
            message=str(exc),
            details=exc.details,
        )
        return JSONResponse(
            status_code=status_code,
            content=err_res.model_dump(mode="json"),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        err_res = APIErrorResponse(
            error_code="VALIDATION_ERROR",
            message=str(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=err_res.model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
        err_res = APIErrorResponse(
            error_code="INTERNAL_ERROR",
            message="An unexpected server error occurred.",
            details={},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=err_res.model_dump(mode="json"),
        )

    # Include master v1 router
    app.include_router(api_v1_router)

    return app


# Default application instance
app = create_app()
