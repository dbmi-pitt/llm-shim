"""Models API endpoint."""

from fastapi import APIRouter, Depends

from llm_shim.api.schemas.openai import ModelListResponse
from llm_shim.services.models import ModelsService, get_models_service

__all__ = ["router"]

router = APIRouter(tags=["models"])


@router.get("/v1/models")
async def list_models(
    service: ModelsService = Depends(get_models_service),
) -> ModelListResponse:
    """List configured chat and embedding model routes in OpenAI format.

    Args:
        service (ModelsService): Injected models service.

    Returns:
        ModelListResponse: OpenAI-compatible model list response.
    """
    return service.list()
