"""Models API endpoint."""

from fastapi import APIRouter

from llm_shim.api.schemas.openai import ModelListResponse
from llm_shim.services.models import get_models_service

__all__ = ["router"]

router = APIRouter(tags=["models"])


@router.get("/v1/models")
async def list_models() -> ModelListResponse:
    """List configured chat and embedding model routes in OpenAI format."""
    return get_models_service().list()
