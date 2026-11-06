"""llm-shim package."""

import asyncio
import logging
import time
from importlib.metadata import version
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from llm_shim.api.chat import router as chat_router
from llm_shim.api.embeddings import router as embeddings_router
from llm_shim.api.models import router as models_router
from llm_shim.core.config import Settings, get_settings
from llm_shim.core.exceptions import (
    AuthenticationError,
    BadRequestError,
    RequestTimeoutError,
    RequestTooLargeError,
    ShimError,
)
from llm_shim.core.security import AccessController, safe_request_id

__all__ = ["app", "create_app"]

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    runtime_settings = settings or get_settings()
    application = FastAPI(
        title="llm-shim",
        version=version("llm-shim"),
        docs_url="/docs" if runtime_settings.security.docs_enabled else None,
        redoc_url="/redoc" if runtime_settings.security.docs_enabled else None,
        openapi_url="/openapi.json" if runtime_settings.security.docs_enabled else None,
    )
    application.state.settings = runtime_settings
    application.state.access_controller = AccessController(runtime_settings.security)

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = safe_request_id(request.headers.get("x-request-id"), uuid4().hex)
        request.state.request_id = request_id
        started = time.monotonic()
        is_v1 = request.url.path.startswith("/v1/")
        controller: AccessController = request.app.state.access_controller
        key_name: str | None = None
        acquired = False
        try:
            if is_v1:
                content_length = request.headers.get("content-length")
                if content_length:
                    try:
                        body_size = int(content_length)
                    except ValueError as error:
                        raise BadRequestError("Invalid Content-Length") from error
                    if body_size > runtime_settings.security.request_body_max_bytes:
                        raise RequestTooLargeError("Request body is too large")
                key_name = controller.authenticate(request.headers.get("authorization"))
                if key_name is None:
                    raise AuthenticationError("Invalid authentication")
                request.state.authenticated_key = key_name
                controller.check_rate_limit(key_name)
                if request.method in {"POST", "PUT", "PATCH"}:
                    _limit_request_body(
                        request, runtime_settings.security.request_body_max_bytes
                    )
                    await controller.acquire()
                    acquired = True
                    try:
                        async with asyncio.timeout(
                            runtime_settings.security.request_timeout_seconds
                        ):
                            response = await call_next(request)
                    except TimeoutError as error:
                        raise RequestTimeoutError("Request timed out") from error
                else:
                    response = await call_next(request)
            else:
                response = await call_next(request)
        except ShimError as exc:
            response = await _error_response(request, exc)
        finally:
            if acquired:
                controller.release()
        response.headers["x-request-id"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        if is_v1:
            logger.info(
                "request_id=%s key=%s endpoint=%s status=%s latency_ms=%d",
                request_id,
                key_name or "unauthenticated",
                request.url.path,
                response.status_code,
                int((time.monotonic() - started) * 1000),
            )
        return response

    @application.exception_handler(ShimError)
    async def shim_error_handler(request: Request, exc: ShimError) -> JSONResponse:
        """Map ShimError subclasses to appropriate HTTP responses."""
        logger.warning(
            "request_id=%s %s: %s",
            getattr(request.state, "request_id", "unknown"),
            type(exc).__name__,
            exc,
        )
        return await _error_response(request, exc)

    @application.get("/livez", include_in_schema=False)
    @application.get("/healthz", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    application.include_router(chat_router)
    application.include_router(embeddings_router)
    application.include_router(models_router)
    return application


async def _error_response(request: Request, exc: ShimError) -> JSONResponse:
    """Log diagnostics internally but keep HTTP errors generic and safe."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning("request_id=%s error=%s", request_id, type(exc).__name__)
    messages = {
        400: "Invalid request.",
        401: "Invalid authentication credentials.",
        413: "Request body is too large.",
        429: "Rate limit exceeded.",
        502: "Upstream provider request failed.",
        503: "Service is temporarily overloaded.",
        504: "Upstream provider request timed out.",
    }
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": messages.get(exc.status_code, "Internal server error."),
                "type": "server_error",
                "request_id": request_id,
            }
        },
    )


def _limit_request_body(request: Request, maximum: int) -> None:
    """Enforce a cumulative limit for chunked bodies without buffering them."""
    receive = request._receive
    consumed = 0

    async def limited_receive():
        nonlocal consumed
        message = await receive()
        if message["type"] == "http.request":
            consumed += len(message.get("body", b""))
            if consumed > maximum:
                raise RequestTooLargeError("Request body is too large")
        return message

    request._receive = limited_receive


app = create_app()
