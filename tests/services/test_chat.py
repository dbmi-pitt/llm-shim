from typing import Any, cast

import pytest
from pydantic import BaseModel

from llm_shim.api.schemas.openai import (
    ChatCompletionRequest,
    ChatMessage,
    ResponseFormatJsonSchema,
    ResponseFormatJsonSchemaDefinition,
)
from llm_shim.services.chat import ChatService


class FakeProvider:
    def __init__(self) -> None:
        self.chat_models = ["gpt-4o-mini"]
        self.embedding_models = []
        self.env = {}
        self.chat_model_settings = {
            "temperature": 0.1,
            "max_tokens": 32,
            "timeout": 30,
        }


class FakeSettings:
    def resolve_chat_provider(
        self, requested_model: str | None
    ) -> tuple[str, str, FakeProvider]:
        del requested_model
        return "openai", "gpt-4o-mini", FakeProvider()


@pytest.mark.asyncio
async def test_chat_service_returns_text_response(monkeypatch: Any) -> None:
    service = ChatService(settings=cast(Any, FakeSettings()))

    async def fake_run_text(
        model_name: str,
        prompt: str,
        model_settings: dict[str, Any] | None,
    ) -> str:
        assert model_name == "openai:gpt-4o-mini"
        assert "user: hello" in prompt
        assert model_settings is not None
        assert model_settings["temperature"] == 0.1
        assert model_settings["max_tokens"] == 32
        assert model_settings["timeout"] == 30
        return "plain completion"

    monkeypatch.setattr(service, "_run_text_model", fake_run_text)

    response = await service.create(
        ChatCompletionRequest(messages=[ChatMessage(role="user", content="hello")])
    )

    assert response.object == "chat.completion"
    assert response.model == "openai:gpt-4o-mini"
    assert response.choices[0].message.content == "plain completion"
    assert response.usage.prompt_tokens == 0
    assert response.usage.completion_tokens == 0
    assert response.usage.total_tokens == 0


@pytest.mark.asyncio
async def test_chat_service_returns_json_schema_response(monkeypatch: Any) -> None:
    service = ChatService(settings=cast(Any, FakeSettings()))

    class Structured(BaseModel):
        answer: str

    async def fake_run_structured(
        model_name: str,
        prompt: str,
        output_model: type[BaseModel],
        model_settings: dict[str, Any] | None,
    ) -> BaseModel:
        assert model_name == "openai:gpt-4o-mini"
        assert "user: hello" in prompt
        assert issubclass(output_model, BaseModel)
        assert model_settings is not None
        return Structured(answer="ok")

    monkeypatch.setattr(service, "_run_structured_model", fake_run_structured)

    response = await service.create(
        ChatCompletionRequest(
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
    assert response.usage.prompt_tokens == 0
    assert response.usage.completion_tokens == 0


@pytest.mark.asyncio
async def test_chat_service_wraps_provider_errors(monkeypatch: Any) -> None:
    service = ChatService(settings=cast(Any, FakeSettings()))

    async def failing_run_text(
        model_name: str,
        prompt: str,
        model_settings: dict[str, Any] | None,
    ) -> str:
        del model_name
        del prompt
        del model_settings
        raise RuntimeError("rate limited")

    monkeypatch.setattr(service, "_run_text_model", failing_run_text)

    with pytest.raises(RuntimeError, match="Provider chat completion failed"):
        await service.create(
            ChatCompletionRequest(messages=[ChatMessage(role="user", content="hello")])
        )


@pytest.mark.asyncio
async def test_request_chat_model_settings_override_provider_defaults(
    monkeypatch: Any,
) -> None:
    service = ChatService(settings=cast(Any, FakeSettings()))

    async def fake_run_text(
        model_name: str,
        prompt: str,
        model_settings: dict[str, Any] | None,
    ) -> str:
        del model_name
        del prompt
        assert model_settings is not None
        assert model_settings["temperature"] == 0.8
        assert model_settings["max_tokens"] == 99
        assert model_settings["timeout"] == 30
        return "ok"

    monkeypatch.setattr(service, "_run_text_model", fake_run_text)

    await service.create(
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hello")],
            temperature=0.8,
            max_tokens=99,
        )
    )


@pytest.mark.asyncio
async def test_chat_service_requires_chat_model() -> None:
    class MissingModelSettings:
        def resolve_chat_provider(
            self, requested_model: str | None
        ) -> tuple[str, str, Any]:
            del requested_model
            raise ValueError(
                "Request model is required when provider chat_models use wildcard patterns"
            )

    service = ChatService(settings=cast(Any, MissingModelSettings()))

    with pytest.raises(ValueError, match="Request model is required"):
        await service.create(
            ChatCompletionRequest(messages=[ChatMessage(role="user", content="hello")])
        )
