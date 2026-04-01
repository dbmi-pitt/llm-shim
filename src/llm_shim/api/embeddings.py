"""Embeddings API endpoint."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from llm_shim.api.models import (
    EmbeddingDatum,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
)
from llm_shim.core.config import get_settings

__all__ = ["router"]

router = APIRouter(tags=["embeddings"])


@router.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingsRequest) -> EmbeddingsResponse:
    """Create OpenAI-compatible embeddings output for providers with embeddings APIs.

    Args:
        request (EmbeddingsRequest): OpenAI embeddings request.

    Returns:
        EmbeddingsResponse: OpenAI embeddings response.
    """
    settings = get_settings()

    try:
        _, provider = settings.resolve_provider(request.model)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    configured_model = str(provider.model)

    try:
        client = provider.create_async_client()
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize provider client: {error}",
        ) from error

    provider_model_name = provider.provider_model_name

    embedding_kwargs = request.provider_create_kwargs(provider_model_name)

    try:
        if client.client is None:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Provider client initialization returned no underlying SDK client"
                ),
            )
        raw_response = await client.client.embeddings.create(**embedding_kwargs)
    except AttributeError as error:
        raise HTTPException(
            status_code=400,
            detail="The selected provider does not expose an embeddings endpoint",
        ) from error
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Provider embeddings request failed: {error}",
        ) from error

    if isinstance(raw_response, BaseModel):
        raw_payload = raw_response.model_dump(mode="json")
    elif isinstance(raw_response, dict):
        raw_payload = raw_response
    else:
        raise HTTPException(
            status_code=502,
            detail="Provider returned an unsupported embeddings response payload",
        )

    data_payload = raw_payload.get("data", [])
    embeddings_data: list[EmbeddingDatum] = []
    if isinstance(data_payload, list):
        for item in data_payload:
            if not isinstance(item, dict):
                continue

            embedding_values = item.get("embedding", [])
            if not isinstance(embedding_values, list):
                continue

            float_embedding: list[float] = []
            for value in embedding_values:
                if isinstance(value, int | float):
                    float_embedding.append(float(value))

            index_value = item.get("index", len(embeddings_data))
            index = (
                index_value if isinstance(index_value, int) else len(embeddings_data)
            )

            embeddings_data.append(
                EmbeddingDatum(index=index, embedding=float_embedding),
            )

    usage_payload = raw_payload.get("usage", {})
    prompt_tokens = 0
    total_tokens = 0
    if isinstance(usage_payload, dict):
        prompt_value = usage_payload.get("prompt_tokens", 0)
        total_value = usage_payload.get("total_tokens", 0)
        prompt_tokens = prompt_value if isinstance(prompt_value, int) else 0
        total_tokens = total_value if isinstance(total_value, int) else 0

    model_value = raw_payload.get("model")
    response_model = model_value if isinstance(model_value, str) else configured_model

    return EmbeddingsResponse(
        data=embeddings_data,
        model=response_model,
        usage=EmbeddingsUsage(prompt_tokens=prompt_tokens, total_tokens=total_tokens),
    )
