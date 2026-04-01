from typing import Any, cast

import pytest
from pydantic import BaseModel

from llm_shim.api.models import (
    ChatCompletionRequest,
    ChatMessage,
    ResponseFormatJsonSchema,
    ResponseFormatJsonSchemaDefinition,
)
from llm_shim.services.chat import ChatService


class FakeStructuredResponse:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def model_dump_json(self) -> str:
        return self._payload


class FakeInstructorClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        response_model = kwargs.get("response_model")
        if response_model is str:
            return "plain completion"
        return FakeStructuredResponse('{"answer":"ok"}')


class FakeProvider:
    def __init__(self, client: FakeInstructorClient) -> None:
        self.model = "openai/gpt-4o-mini"
        self._client = client

    def create_async_client(self) -> FakeInstructorClient:
        return self._client


class FailingProvider:
    model = "openai/gpt-4o-mini"

    def create_async_client(self) -> FakeInstructorClient:
        raise RuntimeError("boom")


class FakeSettings:
    def __init__(self, provider: object) -> None:
        self._provider = provider

    def resolve_provider(self, requested_model: str | None) -> tuple[str, object]:
        del requested_model
        return "default", self._provider


@pytest.mark.asyncio
async def test_chat_service_returns_text_response() -> None:
    client = FakeInstructorClient()
    provider = FakeProvider(client)
    service = ChatService(settings=cast(Any, FakeSettings(provider)))

    response = await service.create(
        ChatCompletionRequest(
            model="default",
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=64,
            temperature=0.2,
            top_p=0.95,
            user="user-1",
        )
    )

    assert response.object == "chat.completion"
    assert response.model == "openai/gpt-4o-mini"
    assert response.choices[0].message.content == "plain completion"
    assert response.usage.prompt_tokens == 0
    assert response.usage.completion_tokens == 0
    assert response.usage.total_tokens == 0
    assert client.calls == [
        {
            "response_model": str,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 64,
            "temperature": 0.2,
            "top_p": 0.95,
            "user": "user-1",
        }
    ]


@pytest.mark.asyncio
async def test_chat_service_returns_json_schema_response() -> None:
    client = FakeInstructorClient()
    provider = FakeProvider(client)
    service = ChatService(settings=cast(Any, FakeSettings(provider)))

    response = await service.create(
        ChatCompletionRequest(
            model="default",
            messages=[ChatMessage(role="user", content="hello")],
            response_format=ResponseFormatJsonSchema(
                type="json_schema",
                json_schema=ResponseFormatJsonSchemaDefinition(
                    name="answer_schema",
                    schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                ),
            ),
        )
    )

    assert response.choices[0].message.content == '{"answer":"ok"}'
    assert len(client.calls) == 1
    response_model = client.calls[0]["response_model"]
    assert isinstance(response_model, type)
    assert issubclass(response_model, BaseModel)


@pytest.mark.asyncio
async def test_chat_service_raises_on_client_init_failure() -> None:
    service = ChatService(settings=cast(Any, FakeSettings(FailingProvider())))

    with pytest.raises(
        RuntimeError, match="Failed to initialize provider client: boom"
    ):
        await service.create(
            ChatCompletionRequest(
                model="default",
                messages=[ChatMessage(role="user", content="hello")],
            )
        )


class FailingInstructorClient:
    """Client that fails when create is called."""

    async def create(self, **kwargs: object) -> object:
        raise RuntimeError("API error: rate limited")


class FailingInstructorClientValueError:
    """Client that fails with ValueError."""

    async def create(self, **kwargs: object) -> object:
        raise ValueError("Invalid schema in response format")


class FailingProviderForCompletion:
    def __init__(self, client: object) -> None:
        self.model = "openai/gpt-4o"
        self._client = client

    def create_async_client(self) -> object:
        return self._client


@pytest.mark.asyncio
async def test_chat_service_raises_on_text_completion_failure() -> None:
    """Chat service should propagate RuntimeError from text completion."""
    client = FailingInstructorClient()
    provider = FailingProviderForCompletion(client)
    service = ChatService(settings=cast(Any, FakeSettings(provider)))

    with pytest.raises(RuntimeError, match="Provider chat completion failed"):
        await service.create(
            ChatCompletionRequest(
                model="default",
                messages=[ChatMessage(role="user", content="hello")],
            )
        )


@pytest.mark.asyncio
async def test_chat_service_raises_on_json_schema_completion_failure() -> None:
    """Chat service should propagate RuntimeError from structured completion."""
    client = FailingInstructorClient()
    provider = FailingProviderForCompletion(client)
    service = ChatService(settings=cast(Any, FakeSettings(provider)))

    with pytest.raises(RuntimeError, match="Provider chat completion failed"):
        await service.create(
            ChatCompletionRequest(
                model="default",
                messages=[ChatMessage(role="user", content="hello")],
                response_format=ResponseFormatJsonSchema(
                    type="json_schema",
                    json_schema=ResponseFormatJsonSchemaDefinition(
                        name="answer_schema",
                        schema={
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                        },
                    ),
                ),
            )
        )


@pytest.mark.asyncio
async def test_chat_service_raises_value_error_from_json_schema() -> None:
    """Chat service should re-raise ValueError from json schema validation."""
    client = FailingInstructorClientValueError()
    provider = FailingProviderForCompletion(client)
    service = ChatService(settings=cast(Any, FakeSettings(provider)))

    with pytest.raises(ValueError, match="Invalid schema in response format"):
        await service.create(
            ChatCompletionRequest(
                model="default",
                messages=[ChatMessage(role="user", content="hello")],
                response_format=ResponseFormatJsonSchema(
                    type="json_schema",
                    json_schema=ResponseFormatJsonSchemaDefinition(
                        name="answer_schema",
                        schema={
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                        },
                    ),
                ),
            )
        )


@pytest.mark.asyncio
async def test_chat_service_builds_correct_response_shape() -> None:
    """Chat service should build response with correct structure."""
    client = FakeInstructorClient()
    provider = FakeProvider(client)
    service = ChatService(settings=cast(Any, FakeSettings(provider)))

    response = await service.create(
        ChatCompletionRequest(
            model="default",
            messages=[ChatMessage(role="user", content="test")],
        )
    )

    # Verify response structure
    assert hasattr(response, "id")
    assert hasattr(response, "created")
    assert len(response.id) > 0
    assert isinstance(response.created, int)
    assert response.created > 0
    assert len(response.choices) == 1
    assert response.choices[0].index == 0
    assert response.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_chat_service_minimal_request() -> None:
    """Chat service should work with minimal request."""
    client = FakeInstructorClient()
    provider = FakeProvider(client)
    service = ChatService(settings=cast(Any, FakeSettings(provider)))

    response = await service.create(
        ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content="You are helpful"),
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi there"),
                ChatMessage(role="user", content="How are you?"),
            ]
        )
    )

    assert response.object == "chat.completion"
    assert len(response.choices) == 1


@pytest.mark.asyncio
async def test_chat_service_multiple_message_types() -> None:
    """Chat service should handle different message roles."""
    client = FakeInstructorClient()
    provider = FakeProvider(client)
    service = ChatService(settings=cast(Any, FakeSettings(provider)))

    await service.create(
        ChatCompletionRequest(
            model="default",
            messages=[
                ChatMessage(role="system", content="system message"),
                ChatMessage(role="user", content="user message"),
                ChatMessage(role="assistant", content="assistant message"),
                ChatMessage(
                    role="tool", content="tool message", tool_call_id="call-123"
                ),
            ],
        )
    )

    assert len(client.calls) == 1
    messages = cast(list[dict[str, object]], client.calls[0]["messages"])
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[3]["role"] == "tool"
    assert messages[3]["tool_call_id"] == "call-123"


def test_get_chat_service_returns_cached_service() -> None:
    """get_chat_service should return the same instance on repeated calls."""
    from llm_shim.services.chat import get_chat_service

    # Clear cache
    get_chat_service.cache_clear()

    service1 = get_chat_service()
    service2 = get_chat_service()

    assert service1 is service2


def test_get_chat_service_returns_chat_service() -> None:
    """get_chat_service should return a ChatService instance."""
    from llm_shim.services.chat import get_chat_service

    get_chat_service.cache_clear()
    service = get_chat_service()

    assert isinstance(service, ChatService)
