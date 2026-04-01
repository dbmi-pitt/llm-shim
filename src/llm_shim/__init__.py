"""llm-shim package."""

from importlib.metadata import version

from fastapi import FastAPI

from llm_shim.api.chat import router as chat_router
from llm_shim.api.embeddings import router as embeddings_router
from llm_shim.api.models import router as models_router

__all__ = ["app", "create_app"]


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application instance.
    """
    application = FastAPI(title="llm-shim", version=version("llm-shim"))

    @application.get("/livez", include_in_schema=False)
    @application.get("/healthz", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    application.include_router(chat_router)
    application.include_router(embeddings_router)
    application.include_router(models_router)
    return application


app = create_app()
