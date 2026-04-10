"""Service for OpenAI-compatible embeddings responses."""

import logging
import os
from functools import lru_cache
from typing import Any, cast

import httpx
from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingSettings
from pydantic_ai.usage import RequestUsage

from llm_shim.api.schemas.openai import (
    EmbeddingDatum,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingsUsage,
)
from llm_shim.core.config import Settings, TeiProviderSettings, get_settings
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
    def _build_tei_inputs(
        inputs: list[str],
        input_prefix_template: str | None,
    ) -> list[str]:
        """Apply optional per-input prompt formatting for TEI requests."""
        if input_prefix_template is None:
            return inputs
        return [input_prefix_template.format(input=value) for value in inputs]

    @staticmethod
    def _build_tei_headers(tei_settings: TeiProviderSettings) -> dict[str, str]:
        """Build HTTP headers for TEI requests."""
        headers = {"Content-Type": "application/json"}
        if tei_settings.auth_token_env is None:
            return headers

        auth_token = os.getenv(tei_settings.auth_token_env)
        if not auth_token:
            raise BadRequestError(
                "TEI auth token env var "
                f"'{tei_settings.auth_token_env}' is not set"
            )
        headers["Authorization"] = f"Bearer {auth_token}"
        return headers

    async def _run_tei_embeddings(
        self,
        tei_settings: TeiProviderSettings,
        inputs: list[str],
    ) -> tuple[list[list[float]], RequestUsage]:
        """Execute embedding generation against a TEI HTTP endpoint."""
        request_inputs = self._build_tei_inputs(
            inputs=inputs,
            input_prefix_template=tei_settings.input_prefix_template,
        )
        base_url = tei_settings.base_url.rstrip("/")
        endpoint = tei_settings.endpoint
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"

        async with httpx.AsyncClient(
            timeout=tei_settings.request_timeout_seconds
        ) as client:
            response = await client.post(
                f"{base_url}{endpoint}",
                headers=self._build_tei_headers(tei_settings),
                json={"inputs": request_inputs},
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            detail = error.response.text.strip()
            message = f"TEI request failed with status {error.response.status_code}"
            if detail:
                message = f"{message}: {detail}"
            raise ProviderCallError(message) from error

        try:
            payload = response.json()
        except ValueError as error:
            raise ProviderCallError("TEI response was not valid JSON") from error

        if not isinstance(payload, list):
            raise ProviderCallError("TEI response must be a JSON array")

        if payload and all(isinstance(value, (int, float)) for value in payload):
            payload = [payload]

        if not all(
            isinstance(vector, list)
            and all(isinstance(value, (int, float)) for value in vector)
            for vector in payload
        ):
            raise ProviderCallError(
                "TEI response must be an embedding vector or list of embedding vectors"
            )

        vectors = [[float(value) for value in vector] for vector in payload]
        return vectors, RequestUsage()

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
                    if provider.backend == "tei":
                        if request.dimensions is not None:
                            raise BadRequestError(
                                "TEI providers do not support request-level dimensions"
                            )
                        if provider.tei is None:
                            raise BadRequestError(
                                "TEI provider is missing tei configuration"
                            )
                        vectors, usage = await self._run_tei_embeddings(
                            tei_settings=provider.tei,
                            inputs=inputs,
                        )
                    else:
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
