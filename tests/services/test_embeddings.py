import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from llm_shim.api.models import EmbeddingsRequest
from llm_shim.services.embeddings import (
    AnthropicEmbeddingsAdapter,
    BedrockEmbeddingsAdapter,
    EmbeddingsService,
    GoogleEmbeddingsAdapter,
    OpenAILikeEmbeddingsAdapter,
)


class FakeOpenAIEmbeddingItem:
    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class FakeOpenAIResponse:
    def __init__(self, items: list[FakeOpenAIEmbeddingItem]) -> None:
        self.data = items


class FakeOpenAIEmbeddingsClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create(
        self,
        *,
        model: str,
        input: list[str] | None = None,
        inputs: list[str] | None = None,
        dimensions: int | None = None,
        encoding_format: str | None = None,
        user: str | None = None,
    ) -> FakeOpenAIResponse:
        self.calls.append(
            {
                "model": model,
                "input": input,
                "inputs": inputs,
                "dimensions": dimensions,
                "encoding_format": encoding_format,
                "user": user,
            },
        )
        return FakeOpenAIResponse(
            [
                FakeOpenAIEmbeddingItem(1, [3.0, 4.0]),
                FakeOpenAIEmbeddingItem(0, [1.0, 2.0]),
            ],
        )


class FakeOpenAINativeClient:
    def __init__(self) -> None:
        self.embeddings = FakeOpenAIEmbeddingsClient()


class FakeGoogleEmbedding:
    def __init__(self, values: list[float]) -> None:
        self.values = values


class FakeGoogleResponse:
    def __init__(self, embeddings: list[FakeGoogleEmbedding]) -> None:
        self.embeddings = embeddings


class FakeGoogleModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def embed_content(
        self,
        *,
        model: str,
        contents: list[str],
        config: dict[str, object] | None = None,
    ) -> FakeGoogleResponse:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return FakeGoogleResponse(
            [
                FakeGoogleEmbedding([0.1, 0.2]),
                FakeGoogleEmbedding([0.3, 0.4]),
            ],
        )


class FakeGoogleAio:
    def __init__(self) -> None:
        self.models = FakeGoogleModels()


class FakeGoogleNativeClient:
    def __init__(self) -> None:
        self.aio = FakeGoogleAio()


class FakeBedrockBody:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _decode_body(body: object) -> dict[str, object]:
    assert isinstance(body, bytes)
    return json.loads(body.decode("utf-8"))


class FakeBedrockClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def invoke_model(self, **kwargs: object) -> dict[str, FakeBedrockBody]:
        self.calls.append(kwargs)
        body = _decode_body(kwargs["body"])
        text = body.get("inputText")
        if text is None:
            texts = body.get("texts")
            assert isinstance(texts, list)
            text = texts[0]
        payload = {"embedding": [float(len(str(text))), float(len(str(text))) + 0.5]}
        return {"body": FakeBedrockBody(payload)}


class FakeProvider:
    def __init__(self, native_client: object, *, provider_name: str = "google") -> None:
        self.model = f"{provider_name}/embed-model"
        self.provider_name = provider_name
        self.provider_model_name = "embed-model"
        self._native_client = native_client

    def create_async_client(self) -> SimpleNamespace:
        return SimpleNamespace(client=self._native_client)


class FakeSettings:
    def __init__(self, provider: object) -> None:
        self._provider = provider

    def resolve_provider(self, requested_model: str | None) -> tuple[str, object]:
        del requested_model
        return "default", self._provider


@pytest.mark.asyncio
async def test_openai_like_adapter_sorts_vectors_by_index() -> None:
    client = FakeOpenAINativeClient()
    adapter = OpenAILikeEmbeddingsAdapter()
    request = EmbeddingsRequest(input=["alpha", "beta"], dimensions=8, user="user-1")

    vectors = await adapter.create_vectors(
        client,
        "mistral-embed",
        request,
    )

    assert vectors == [[1.0, 2.0], [3.0, 4.0]]
    assert client.embeddings.calls == [
        {
            "model": "mistral-embed",
            "input": ["alpha", "beta"],
            "inputs": None,
            "dimensions": 8,
            "encoding_format": None,
            "user": "user-1",
        }
    ]


@pytest.mark.asyncio
async def test_google_adapter_uses_output_dimensionality() -> None:
    client = FakeGoogleNativeClient()
    adapter = GoogleEmbeddingsAdapter()
    request = EmbeddingsRequest(input=["alpha", "beta"], dimensions=16)

    vectors = await adapter.create_vectors(
        client,
        "gemini-embedding-001",
        request,
    )

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert client.aio.models.calls == [
        {
            "model": "gemini-embedding-001",
            "contents": ["alpha", "beta"],
            "config": {"output_dimensionality": 16},
        }
    ]


@pytest.mark.asyncio
async def test_bedrock_adapter_invokes_model_per_input() -> None:
    client = FakeBedrockClient()
    adapter = BedrockEmbeddingsAdapter()
    request = EmbeddingsRequest(input=["alpha", "beta"], dimensions=1024)

    vectors = await adapter.create_vectors(
        client,
        "amazon.titan-embed-text-v2:0",
        request,
    )

    assert vectors == [[5.0, 5.5], [4.0, 4.5]]
    assert [_decode_body(call["body"]) for call in client.calls] == [
        {"inputText": "alpha", "dimensions": 1024},
        {"inputText": "beta", "dimensions": 1024},
    ]


@pytest.mark.asyncio
async def test_service_builds_openai_response_shape() -> None:
    provider = FakeProvider(FakeGoogleNativeClient(), provider_name="google")
    service = EmbeddingsService(settings=cast(Any, FakeSettings(provider)))

    response = await service.create(EmbeddingsRequest(input=["alpha", "beta"]))

    assert response.object == "list"
    assert response.model == "google/embed-model"
    assert response.usage.prompt_tokens == 0
    assert response.usage.total_tokens == 0
    assert [item.index for item in response.data] == [0, 1]
    assert response.data[0].embedding == [0.1, 0.2]


class FailingProvider:
    """Provider that fails to create a client."""

    def __init__(self) -> None:
        self.model = "google/embed-model"
        self.provider_name = "google"
        self.provider_model_name = "embed-model"

    def create_async_client(self) -> None:
        raise RuntimeError("API key invalid")


class FailingSettings:
    def __init__(self, provider: object) -> None:
        self._provider = provider

    def resolve_provider(self, requested_model: str | None) -> tuple[str, object]:
        del requested_model
        return "default", self._provider


@pytest.mark.asyncio
async def test_embeddings_service_raises_on_client_init_failure() -> None:
    """EmbeddingsService should propagate client init failures."""
    service = EmbeddingsService(settings=cast(Any, FailingSettings(FailingProvider())))

    with pytest.raises(RuntimeError, match="Failed to initialize provider client"):
        await service.create(EmbeddingsRequest(input="hello"))


class MissingEmbeddingsAttributeClient:
    """Client without embeddings attribute."""

    pass


@pytest.mark.asyncio
async def test_openai_adapter_raises_on_missing_embeddings_attribute() -> None:
    """OpenAI adapter should raise if client lacks embeddings attribute."""
    adapter = OpenAILikeEmbeddingsAdapter()
    request = EmbeddingsRequest(input="hello")

    with pytest.raises(ValueError, match=r"does not expose an embeddings\.create"):
        await adapter.create_vectors(
            MissingEmbeddingsAttributeClient(), "model", request
        )


class BrokenEmbeddingsClient:
    """Client with broken embeddings endpoint."""

    embeddings = None


@pytest.mark.asyncio
async def test_openai_adapter_raises_on_missing_create_method() -> None:
    """OpenAI adapter should raise if embeddings lacks create method."""
    adapter = OpenAILikeEmbeddingsAdapter()
    request = EmbeddingsRequest(input="hello")

    with pytest.raises(ValueError, match=r"does not expose an embeddings\.create"):
        await adapter.create_vectors(BrokenEmbeddingsClient(), "model", request)


class WrongSignatureEmbeddingsClient:
    """Client with wrong embeddings.create signature."""

    class Embeddings:
        async def create(self, *, model: str) -> object:  # missing input/inputs
            return object()

    embeddings = Embeddings()


@pytest.mark.asyncio
async def test_openai_adapter_raises_on_unsupported_signature() -> None:
    """OpenAI adapter should raise if create doesn't support input payloads."""
    adapter = OpenAILikeEmbeddingsAdapter()
    request = EmbeddingsRequest(input="hello")

    with pytest.raises(ValueError, match="does not accept input payloads"):
        await adapter.create_vectors(WrongSignatureEmbeddingsClient(), "model", request)


class NoDataEmbeddingsResponse:
    """Response without data attribute."""

    pass


class ProviderWithDatalessResponse:
    """Client returning response without data."""

    class Embeddings:
        async def create(self, **kwargs: object) -> NoDataEmbeddingsResponse:
            return NoDataEmbeddingsResponse()

    embeddings = Embeddings()


@pytest.mark.asyncio
async def test_openai_adapter_raises_on_missing_data() -> None:
    """OpenAI adapter should raise if response lacks data."""

    # Create a proper response without data attribute
    class BadResponse:
        pass

    class ProperEmbeddingsClient:
        async def create(self, *, model: str, input: list[str]) -> BadResponse:
            return BadResponse()

    class ProperProvider:
        embeddings = SimpleNamespace(create=ProperEmbeddingsClient().create)

    adapter = OpenAILikeEmbeddingsAdapter()
    request = EmbeddingsRequest(input="hello")

    with pytest.raises(ValueError, match="did not include data"):
        await adapter.create_vectors(ProperProvider(), "model", request)


@pytest.mark.asyncio
async def test_openai_adapter_uses_inputs_parameter() -> None:
    """OpenAI adapter should use 'inputs' if 'input' not available."""

    class InputsEmbeddingsClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def create(
            self,
            *,
            model: str,
            inputs: list[str] | None = None,
        ) -> FakeOpenAIResponse:
            self.calls.append({"model": model, "inputs": inputs})
            return FakeOpenAIResponse(
                [
                    FakeOpenAIEmbeddingItem(0, [1.0, 2.0]),
                ]
            )

    class InputsEmbeddingsProvider:
        embeddings: object

        def __init__(self) -> None:
            self.embeddings = InputsEmbeddingsClient()

    adapter = OpenAILikeEmbeddingsAdapter()
    request = EmbeddingsRequest(input="hello")

    vectors = await adapter.create_vectors(InputsEmbeddingsProvider(), "model", request)

    assert len(vectors) == 1


@pytest.mark.asyncio
async def test_google_adapter_without_dimensions() -> None:
    """Google adapter should not include dimensions if not specified."""
    client = FakeGoogleNativeClient()
    adapter = GoogleEmbeddingsAdapter()
    request = EmbeddingsRequest(input=["alpha", "beta"])

    vectors = await adapter.create_vectors(client, "gemini-embedding-001", request)

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert client.aio.models.calls[0]["config"] is None


class GoogleResponseWithoutEmbeddings:
    """Google response without embeddings."""

    pass


class GoogleClientWithoutEmbeddings:
    def __init__(self) -> None:
        self.aio = SimpleNamespace(models=SimpleNamespace())

    async def embed_content(self, **kwargs: object) -> GoogleResponseWithoutEmbeddings:
        return GoogleResponseWithoutEmbeddings()


class FakeGoogleAioMissing:
    def __init__(self) -> None:
        self.models = SimpleNamespace()

    async def embed_content(self, **kwargs: object) -> FakeGoogleResponse:
        return FakeGoogleResponse([])


@pytest.mark.asyncio
async def test_google_adapter_handles_empty_embeddings() -> None:
    """Google adapter should handle empty embeddings."""
    client = SimpleNamespace(aio=FakeGoogleAio())
    adapter = GoogleEmbeddingsAdapter()
    request = EmbeddingsRequest(input=["test"])

    # Patch the response to return empty embeddings
    async def mock_embed(
        model: str, contents: list[str], config: object
    ) -> FakeGoogleResponse:
        return FakeGoogleResponse([])

    client.aio.models.embed_content = mock_embed

    vectors = await adapter.create_vectors(client, "model", request)
    assert vectors == []


class BedrockClientCohere:
    """Bedrock client for Cohere model."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def invoke_model(self, **kwargs: object) -> dict[str, FakeBedrockBody]:
        self.calls.append(kwargs)
        payload = {"embeddings": [[1.0, 2.0], [3.0, 4.0]]}
        return {"body": FakeBedrockBody(payload)}


@pytest.mark.asyncio
async def test_bedrock_adapter_handles_cohere_format() -> None:
    """Bedrock adapter should handle Cohere's embeddings format."""
    client = BedrockClientCohere()
    adapter = BedrockEmbeddingsAdapter()
    request = EmbeddingsRequest(input=["test", "data"], dimensions=256)

    vectors = await adapter.create_vectors(client, "cohere.embed-english-v3", request)

    # Cohere adapter returns flat embeddings list, one per input
    assert len(vectors) == 2
    assert vectors[0] == [1.0, 2.0]
    body = _decode_body(client.calls[0]["body"])
    assert body["input_type"] == "search_document"
    assert "texts" in body


@pytest.mark.asyncio
async def test_bedrock_adapter_handles_cohere_nested_format() -> None:
    """Bedrock adapter should handle Cohere's nested embeddings response."""

    class CohereNestedClient:
        def invoke_model(self, **kwargs: object) -> dict[str, FakeBedrockBody]:
            payload = {
                "embeddings": [
                    {"embedding": [1.0, 2.0]},
                    {"embedding": [3.0, 4.0]},
                ]
            }
            return {"body": FakeBedrockBody(payload)}

    adapter = BedrockEmbeddingsAdapter()
    request = EmbeddingsRequest(input=["a", "b"])

    vectors = await adapter.create_vectors(
        CohereNestedClient(), "cohere.embed", request
    )

    assert len(vectors) == 2
    assert vectors[0] == [1.0, 2.0]
    assert vectors[1] == [1.0, 2.0]


class BedrockClientMissingEmbedding:
    """Bedrock response without embeddings."""

    def invoke_model(self, **kwargs: object) -> dict[str, FakeBedrockBody]:
        payload = {}
        return {"body": FakeBedrockBody(payload)}


@pytest.mark.asyncio
async def test_bedrock_adapter_raises_on_missing_embedding() -> None:
    """Bedrock adapter should raise if response lacks embedding."""
    adapter = BedrockEmbeddingsAdapter()
    request = EmbeddingsRequest(input="test")

    with pytest.raises(ValueError, match="did not include embeddings"):
        await adapter.create_vectors(BedrockClientMissingEmbedding(), "model", request)


@pytest.mark.asyncio
async def test_anthropic_adapter_raises_not_supported() -> None:
    """Anthropic adapter should raise because no embeddings API."""
    adapter = AnthropicEmbeddingsAdapter()
    request = EmbeddingsRequest(input="hello")

    with pytest.raises(ValueError, match="does not expose an embeddings API"):
        await adapter.create_vectors(None, "model", request)


@pytest.mark.asyncio
async def test_embeddings_adapter_inputs_helper() -> None:
    """_inputs helper should convert string to list."""
    adapter = OpenAILikeEmbeddingsAdapter()

    # Test single string
    result = adapter._inputs(EmbeddingsRequest(input="hello"))
    assert result == ["hello"]

    # Test list
    result = adapter._inputs(EmbeddingsRequest(input=["hello", "world"]))
    assert result == ["hello", "world"]


@pytest.mark.asyncio
async def test_embeddings_service_with_anthropic_provider() -> None:
    """EmbeddingsService should use correct adapter for anthropic."""
    anthropic_client = MissingEmbeddingsAttributeClient()

    class AnthropicProvider:
        model = "anthropic/some-model"
        provider_name = "anthropic"
        provider_model_name = "some-model"

        def create_async_client(self) -> object:
            return SimpleNamespace(client=anthropic_client)

    settings = cast(Any, FakeSettings(AnthropicProvider()))
    service = EmbeddingsService(settings=settings)

    with pytest.raises(ValueError, match="does not expose an embeddings API"):
        await service.create(EmbeddingsRequest(input="hello"))


@pytest.mark.asyncio
async def test_embeddings_service_propagates_service_errors() -> None:
    """EmbeddingsService should propagate service layer errors."""

    class FailingEmbeddingsClient:
        async def embed_content(self, **kwargs: object) -> None:
            raise RuntimeError("Service error")

    class FailingProvider:
        model = "google/model"
        provider_name = "google"
        provider_model_name = "model"

        def create_async_client(self) -> object:
            return SimpleNamespace(
                client=SimpleNamespace(
                    aio=SimpleNamespace(models=FailingEmbeddingsClient())
                )
            )

    settings = cast(Any, FakeSettings(FailingProvider()))
    service = EmbeddingsService(settings=settings)

    with pytest.raises(RuntimeError, match="Provider embeddings failed"):
        await service.create(EmbeddingsRequest(input="hello"))


@pytest.mark.asyncio
async def test_openai_adapter_with_encoding_format() -> None:
    """OpenAI adapter should pass encoding_format when supported."""

    class EncodingFormatClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def create(
            self,
            *,
            model: str,
            input: list[str],
            encoding_format: str | None = None,
        ) -> FakeOpenAIResponse:
            self.calls.append(
                {"model": model, "input": input, "encoding_format": encoding_format}
            )
            return FakeOpenAIResponse([FakeOpenAIEmbeddingItem(0, [1.0])])

    class EncodingFormatProvider:
        embeddings = EncodingFormatClient()

    adapter = OpenAILikeEmbeddingsAdapter()
    request = EmbeddingsRequest(input="hello", encoding_format="float")

    vectors = await adapter.create_vectors(EncodingFormatProvider(), "model", request)

    assert len(vectors) == 1
    assert EncodingFormatProvider.embeddings.calls[0]["encoding_format"] == "float"


def test_get_embeddings_service_returns_cached_service() -> None:
    """get_embeddings_service should return the same instance on repeated calls."""
    from llm_shim.services.embeddings import get_embeddings_service

    # Clear cache
    get_embeddings_service.cache_clear()

    service1 = get_embeddings_service()
    service2 = get_embeddings_service()

    assert service1 is service2


def test_get_embeddings_service_returns_embeddings_service() -> None:
    """get_embeddings_service should return an EmbeddingsService instance."""
    from llm_shim.services.embeddings import get_embeddings_service

    get_embeddings_service.cache_clear()
    service = get_embeddings_service()

    assert isinstance(service, EmbeddingsService)
