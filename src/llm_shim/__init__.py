"""llm-shim package."""

import logging
from importlib.metadata import version

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from llm_shim.api.chat import router as chat_router
from llm_shim.api.embeddings import router as embeddings_router
from llm_shim.api.models import router as models_router
from llm_shim.core.exceptions import ShimError

__all__ = ["app", "create_app"]

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    application = FastAPI(title="llm-shim", version=version("llm-shim"))

    @application.exception_handler(ShimError)
    async def shim_error_handler(request: Request, exc: ShimError) -> JSONResponse:
        """Map ShimError subclasses to appropriate HTTP responses."""
        logger.warning(
            "request_id=%s %s: %s",
            getattr(request.state, "request_id", "unknown"),
            type(exc).__name__,
            exc,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": str(exc)},
        )

    @application.get("/livez", include_in_schema=False)
    @application.get("/healthz", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    application.include_router(chat_router)
    application.include_router(embeddings_router)
    application.include_router(models_router)
    return application


app = create_app()
