"""Embeddings API endpoint."""

from fastapi import APIRouter, HTTPException

from llm_shim.api.schemas.openai import EmbeddingsRequest, EmbeddingsResponse
from llm_shim.services.embeddings import EmbeddingsService

__all__ = ["router"]

router = APIRouter(tags=["embeddings"])


@router.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingsRequest) -> EmbeddingsResponse:
    """Create OpenAI-compatible embeddings output for providers with embeddings APIs.

    Args:
        request (EmbeddingsRequest): OpenAI embeddings request.
        service (EmbeddingsService): Injected embeddings service.

    Returns:
        EmbeddingsResponse: OpenAI embeddings response.
    """
    service = EmbeddingsService()
    try:
        return await service.create(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        detail = str(error)
        if detail.startswith("Failed to initialize provider client:"):
            raise HTTPException(status_code=500, detail=detail) from error
        raise HTTPException(status_code=502, detail=detail) from error
