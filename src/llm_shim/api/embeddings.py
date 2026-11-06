"""Embeddings API endpoint."""

from fastapi import APIRouter, Depends, Request

from llm_shim.api.schemas.openai import EmbeddingsRequest, EmbeddingsResponse
from llm_shim.api.validation import validate_embeddings_request
from llm_shim.services.embeddings import EmbeddingsService, get_embeddings_service

__all__ = ["router"]

router = APIRouter(tags=["embeddings"])


@router.post("/v1/embeddings")
async def create_embeddings(
    request: EmbeddingsRequest,
    http_request: Request,
    service: EmbeddingsService = Depends(get_embeddings_service),
) -> EmbeddingsResponse:
    """Create OpenAI-compatible embeddings output for providers with embeddings APIs.

    Args:
        request (EmbeddingsRequest): OpenAI embeddings request.
        service (EmbeddingsService): Injected embeddings service.
        http_request (Request): HTTP request used for configured limits.

    Returns:
        EmbeddingsResponse: OpenAI embeddings response.
    """
    validate_embeddings_request(request, http_request.app.state.settings.security)
    return await service.create(request)
