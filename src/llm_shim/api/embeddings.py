"""Embeddings API endpoint."""

from fastapi import APIRouter, Depends

from llm_shim.api.schemas.openai import EmbeddingsRequest, EmbeddingsResponse
from llm_shim.services.embeddings import EmbeddingsService, get_embeddings_service

__all__ = ["router"]

router = APIRouter(tags=["embeddings"])


@router.post("/v1/embeddings")
async def create_embeddings(
    request: EmbeddingsRequest,
    service: EmbeddingsService = Depends(get_embeddings_service),
) -> EmbeddingsResponse:
    """Create OpenAI-compatible embeddings output for providers with embeddings APIs.

    Args:
        request (EmbeddingsRequest): OpenAI embeddings request.
        service (EmbeddingsService): Injected embeddings service.

    Returns:
        EmbeddingsResponse: OpenAI embeddings response.
    """
    return await service.create(request)
