"""Service for OpenAI-compatible embeddings responses."""

import logging
from functools import lru_cache
from typing import Any, cast

from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingSettings
from pydantic_ai.usage import RequestUsage

from llm_shim.api.schemas.openai import (
    EmbeddingDatum,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
)
from llm_shim.core.config import Settings, get_settings
from llm_shim.core.exceptions import BadRequestError, ProviderCallError
from llm_shim.core.utils import environment_override_lock, patched_environ

logger = logging.getLogger(__name__)


class EmbeddingsService:
    """Service for creating OpenAI-compatible embeddings responses."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize service dependencies."""
        self._settings = settings or get_settings()

    def _build_response(
        self,
        configured_model: str,
        vectors: list[list[float]],
        usage: RequestUsage,
    ) -> EmbeddingsResponse:
        """Build an OpenAI-compatible response from provider embedding vectors."""
        return EmbeddingsResponse(
            data=[
                EmbeddingDatum(index=index, embedding=list(vector))
                for index, vector in enumerate(vectors)
            ],
            model=configured_model,
            usage=EmbeddingsUsage(
                prompt_tokens=usage.input_tokens,
                total_tokens=usage.input_tokens,
            ),
        )

    async def _run_embeddings(
        self,
        model_name: str,
        inputs: list[str],
        model_settings: EmbeddingSettings | None,
    ) -> tuple[list[list[float]], RequestUsage]:
        """Execute embedding generation with pydantic-ai."""
        result = await Embedder(model_name).embed_documents(
            inputs,
            settings=model_settings,
        )
        vectors = [list(vector) for vector in result.embeddings]
        return vectors, result.usage

    @staticmethod
    def _build_embedding_settings(
        provider_settings: dict[str, Any],
        dimensions: int | None,
    ) -> EmbeddingSettings | None:
        """Merge provider defaults with request-level embedding settings."""
        allowed = {"dimensions", "truncate", "extra_headers", "extra_body"}
        merged: dict[str, Any] = {
            key: value for key, value in provider_settings.items() if key in allowed
        }
        if dimensions is not None:
            merged["dimensions"] = dimensions
        return cast(EmbeddingSettings, merged) if merged else None

    async def create(self, request: EmbeddingsRequest) -> EmbeddingsResponse:
        """Create provider embeddings and normalize into OpenAI response format.

        Args:
            request (EmbeddingsRequest): OpenAI-compatible embeddings request.

        Returns:
            EmbeddingsResponse: OpenAI-compatible embeddings response.
        """
        provider_id, resolved_model, provider = (
            self._settings.resolve_embedding_provider(request.model)
        )
        configured_model = f"{provider_id}:{resolved_model}"
        model_settings = self._build_embedding_settings(
            provider_settings=provider.embedding_model_settings,
            dimensions=request.dimensions,
        )

        inputs = [request.input] if isinstance(request.input, str) else request.input

        try:
            async with environment_override_lock:
                with patched_environ(provider.env):
                    vectors, usage = await self._run_embeddings(
                        model_name=configured_model,
                        inputs=inputs,
                        model_settings=model_settings,
                    )
        except BadRequestError, ProviderCallError:
            raise
        except Exception as error:
            logger.exception("Embeddings failed for %s", configured_model)
            raise ProviderCallError(f"Provider embeddings failed: {error}") from error

        logger.info(
            "Embeddings for %s: %d input tokens",
            configured_model,
            usage.input_tokens,
        )
        return self._build_response(
            configured_model=configured_model,
            vectors=vectors,
            usage=usage,
        )


@lru_cache(maxsize=1)
def get_embeddings_service() -> EmbeddingsService:
    """Singleton factory for EmbeddingsService.

    Returns:
        EmbeddingsService: Cached service instance.
    """
    return EmbeddingsService()
