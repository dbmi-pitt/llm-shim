from typing import Any, cast

import httpx
import pytest
from pydantic_ai.usage import RequestUsage

from llm_shim.api.schemas.openai import EmbeddingsRequest
from llm_shim.core.config import TeiProviderSettings
from llm_shim.core.exceptions import BadRequestError, ProviderCallError
from llm_shim.services.embeddings import EmbeddingsService


class FakeProvider:
    def __init__(self) -> None:
        self.chat_models = []
        self.embedding_models = ["text-embedding-3-small"]
        self.backend = "pydantic_ai"
        self.env = {}
        self.embedding_model_settings = {
            "dimensions": 64,
            "truncate": True,
        }
        self.tei = None


class FakeTeiProvider:
    def __init__(self) -> None:
        self.chat_models = []
        self.embedding_models = ["Qwen/Qwen3-Embedding-0.6B"]
        self.backend = "tei"
        self.env = {"TEI_API_TOKEN": "secret-token"}
        self.embedding_model_settings = {}
        self.tei = TeiProviderSettings(
            base_url="http://tei.internal:8080",
            endpoint="/embed",
            auth_token_env="TEI_API_TOKEN",
            input_prefix_template=(
                "Instruct: Given a web search query, retrieve relevant passages "
                "that answer the query\nQuery: {input}"
            ),
        )


class FakeSettings:
    def resolve_embedding_provider(
        self, requested_model: str | None
    ) -> tuple[str, str, FakeProvider]:
        assert requested_model == "openai:text-embedding-3-small"
        return "openai", "text-embedding-3-small", FakeProvider()


class FakeTeiSettings:
    def resolve_embedding_provider(
        self, requested_model: str | None
    ) -> tuple[str, str, FakeTeiProvider]:
        assert requested_model == "tei-qwen:Qwen/Qwen3-Embedding-0.6B"
        return "tei-qwen", "Qwen/Qwen3-Embedding-0.6B", FakeTeiProvider()


@pytest.mark.asyncio
async def test_embeddings_service_builds_openai_shape(monkeypatch: Any) -> None:
    service = EmbeddingsService(settings=cast(Any, FakeSettings()))

    async def fake_run_embeddings(
        model_name: str,
        inputs: list[str],
        model_settings: dict[str, Any] | None,
    ) -> tuple[list[list[float]], RequestUsage]:
        assert model_name == "openai:text-embedding-3-small"
        assert inputs == ["alpha", "beta"]
        assert model_settings is not None
        assert model_settings["dimensions"] == 8
        assert model_settings["truncate"] is True
        return [[1.0, 2.0], [3.0, 4.0]], RequestUsage(input_tokens=12)

    monkeypatch.setattr(service, "_run_embeddings", fake_run_embeddings)

    response = await service.create(
        EmbeddingsRequest(
            model="openai:text-embedding-3-small",
            input=["alpha", "beta"],
            dimensions=8,
        )
    )

    assert response.object == "list"
    assert response.model == "openai:text-embedding-3-small"
    assert [item.index for item in response.data] == [0, 1]
    assert response.data[0].embedding == [1.0, 2.0]
    assert response.usage.prompt_tokens == 12
    assert response.usage.total_tokens == 12


@pytest.mark.asyncio
async def test_embeddings_service_wraps_provider_errors(monkeypatch: Any) -> None:
    service = EmbeddingsService(settings=cast(Any, FakeSettings()))

    async def failing_run_embeddings(
        model_name: str,
        inputs: list[str],
        model_settings: dict[str, Any] | None,
    ) -> tuple[list[list[float]], RequestUsage]:
        del model_name
        del inputs
        del model_settings
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(service, "_run_embeddings", failing_run_embeddings)

    with pytest.raises(ProviderCallError, match="Provider embeddings failed"):
        await service.create(
            EmbeddingsRequest(model="openai:text-embedding-3-small", input="hello")
        )


@pytest.mark.asyncio
async def test_embedding_dimensions_override_provider_defaults(
    monkeypatch: Any,
) -> None:
    service = EmbeddingsService(settings=cast(Any, FakeSettings()))

    async def fake_run_embeddings(
        model_name: str,
        inputs: list[str],
        model_settings: dict[str, Any] | None,
    ) -> tuple[list[list[float]], RequestUsage]:
        del model_name
        del inputs
        assert model_settings is not None
        assert model_settings["dimensions"] == 8
        assert model_settings["truncate"] is True
        return [[1.0, 2.0]], RequestUsage()

    monkeypatch.setattr(service, "_run_embeddings", fake_run_embeddings)

    await service.create(
        EmbeddingsRequest(
            model="openai:text-embedding-3-small",
            input="alpha",
            dimensions=8,
        )
    )


@pytest.mark.asyncio
async def test_embeddings_service_requires_embedding_model() -> None:
    class MissingModelSettings:
        def resolve_embedding_provider(
            self, requested_model: str | None
        ) -> tuple[str, str, Any]:
            del requested_model
            raise BadRequestError(
                "Request model is required and must use provider:model format"
            )

    service = EmbeddingsService(settings=cast(Any, MissingModelSettings()))

    with pytest.raises(BadRequestError, match="Request model is required"):
        await service.create(EmbeddingsRequest(input="hello"))


@pytest.mark.asyncio
async def test_embeddings_service_calls_tei_endpoint(monkeypatch: Any) -> None:
    service = EmbeddingsService(settings=cast(Any, FakeTeiSettings()))
    captured: dict[str, Any] = {}

    class FakeResponse:
        text = ""
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[list[float]]:
            return [[0.1, 0.2], [0.3, 0.4]]

    class FakeAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            captured["timeout"] = timeout

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            del exc_type
            del exc
            del tb

        async def post(
            self,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, list[str]],
        ) -> FakeResponse:
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = await service.create(
        EmbeddingsRequest(
            model="tei-qwen:Qwen/Qwen3-Embedding-0.6B",
            input=["What is the capital of China?", "Explain gravity"],
        )
    )

    assert captured["timeout"] == 30.0
    assert captured["url"] == "http://tei.internal:8080/embed"
    assert captured["headers"] == {
        "Content-Type": "application/json",
        "Authorization": "Bearer secret-token",
    }
    assert captured["json"] == {
        "inputs": [
            "Instruct: Given a web search query, retrieve relevant passages that "
            "answer the query\nQuery: What is the capital of China?",
            "Instruct: Given a web search query, retrieve relevant passages that "
            "answer the query\nQuery: Explain gravity",
        ]
    }
    assert response.model == "tei-qwen:Qwen/Qwen3-Embedding-0.6B"
    assert response.data[0].embedding == [0.1, 0.2]
    assert response.usage.prompt_tokens == 0


@pytest.mark.asyncio
async def test_embeddings_service_rejects_dimensions_for_tei() -> None:
    service = EmbeddingsService(settings=cast(Any, FakeTeiSettings()))

    with pytest.raises(
        BadRequestError,
        match="do not support request-level dimensions",
    ):
        await service.create(
            EmbeddingsRequest(
                model="tei-qwen:Qwen/Qwen3-Embedding-0.6B",
                input="Explain gravity",
                dimensions=128,
            )
        )


@pytest.mark.asyncio
async def test_embeddings_service_wraps_tei_http_errors(monkeypatch: Any) -> None:
    service = EmbeddingsService(settings=cast(Any, FakeTeiSettings()))

    request = httpx.Request("POST", "http://tei.internal:8080/embed")
    response = httpx.Response(503, request=request, text="upstream unavailable")

    class FakeAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            del timeout

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            del exc_type
            del exc
            del tb

        async def post(
            self,
            url: str,
            *,
            headers: dict[str, str],
            json: dict[str, list[str]],
        ) -> httpx.Response:
            del url
            del headers
            del json
            return response

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    with pytest.raises(ProviderCallError, match="TEI request failed with status 503"):
        await service.create(
            EmbeddingsRequest(
                model="tei-qwen:Qwen/Qwen3-Embedding-0.6B",
                input="Explain gravity",
            )
        )
