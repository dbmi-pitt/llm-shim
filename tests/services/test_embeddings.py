from typing import Any, cast

import pytest

from llm_shim.api.models import EmbeddingsRequest
from llm_shim.services.embeddings import EmbeddingsService


class FakeProvider:
    def __init__(self) -> None:
        self.chat_models = []
        self.embedding_models = ["text-embedding-3-small"]
        self.env = {}
        self.embedding_model_settings = {
            "dimensions": 64,
            "truncate": True,
        }


class FakeSettings:
    def resolve_embedding_provider(
        self, requested_model: str | None
    ) -> tuple[str, str, FakeProvider]:
        del requested_model
        return "openai", "text-embedding-3-small", FakeProvider()


@pytest.mark.asyncio
async def test_embeddings_service_builds_openai_shape(monkeypatch: Any) -> None:
    service = EmbeddingsService(settings=cast(Any, FakeSettings()))

    async def fake_run_embeddings(
        model_name: str,
        inputs: list[str],
        model_settings: dict[str, Any] | None,
    ) -> list[list[float]]:
        assert model_name == "openai:text-embedding-3-small"
        assert inputs == ["alpha", "beta"]
        assert model_settings is not None
        assert model_settings["dimensions"] == 8
        assert model_settings["truncate"] is True
        return [[1.0, 2.0], [3.0, 4.0]]

    monkeypatch.setattr(service, "_run_embeddings", fake_run_embeddings)

    response = await service.create(
        EmbeddingsRequest(input=["alpha", "beta"], dimensions=8)
    )

    assert response.object == "list"
    assert response.model == "openai:text-embedding-3-small"
    assert [item.index for item in response.data] == [0, 1]
    assert response.data[0].embedding == [1.0, 2.0]
    assert response.usage.prompt_tokens == 0
    assert response.usage.total_tokens == 0


@pytest.mark.asyncio
async def test_embeddings_service_wraps_provider_errors(monkeypatch: Any) -> None:
    service = EmbeddingsService(settings=cast(Any, FakeSettings()))

    async def failing_run_embeddings(
        model_name: str,
        inputs: list[str],
        model_settings: dict[str, Any] | None,
    ) -> list[list[float]]:
        del model_name
        del inputs
        del model_settings
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(service, "_run_embeddings", failing_run_embeddings)

    with pytest.raises(RuntimeError, match="Provider embeddings failed"):
        await service.create(EmbeddingsRequest(input="hello"))


@pytest.mark.asyncio
async def test_embedding_dimensions_override_provider_defaults(
    monkeypatch: Any,
) -> None:
    service = EmbeddingsService(settings=cast(Any, FakeSettings()))

    async def fake_run_embeddings(
        model_name: str,
        inputs: list[str],
        model_settings: dict[str, Any] | None,
    ) -> list[list[float]]:
        del model_name
        del inputs
        assert model_settings is not None
        assert model_settings["dimensions"] == 8
        assert model_settings["truncate"] is True
        return [[1.0, 2.0]]

    monkeypatch.setattr(service, "_run_embeddings", fake_run_embeddings)

    await service.create(EmbeddingsRequest(input="alpha", dimensions=8))


@pytest.mark.asyncio
async def test_embeddings_service_requires_embedding_model() -> None:
    class MissingModelSettings:
        def resolve_embedding_provider(
            self, requested_model: str | None
        ) -> tuple[str, str, Any]:
            del requested_model
            raise ValueError(
                "Request model is required when provider embedding_models use wildcard patterns"
            )

    service = EmbeddingsService(settings=cast(Any, MissingModelSettings()))

    with pytest.raises(ValueError, match="Request model is required"):
        await service.create(EmbeddingsRequest(input="hello"))
