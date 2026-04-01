"""Tests for API models and schema handling."""

from typing import Any, cast

import pytest
from pydantic import BaseModel, ValidationError

from llm_shim.api.models import (
    ChatCompletionRequest,
    ChatMessage,
    EmbeddingsRequest,
    JsonSchemaModelFactory,
    ResponseFormatJsonSchema,
    ResponseFormatJsonSchemaDefinition,
)


def test_chat_message_all_fields() -> None:
    """ChatMessage should accept all optional fields."""
    msg = ChatMessage(
        role="user",
        content="hello",
        name="test-user",
        tool_call_id="call-123",
    )
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.name == "test-user"
    assert msg.tool_call_id == "call-123"


def test_chat_message_required_fields() -> None:
    """ChatMessage should require role and content."""
    msg = ChatMessage(role="assistant", content="response")
    assert msg.name is None
    assert msg.tool_call_id is None


def test_chat_message_invalid_role() -> None:
    """ChatMessage should validate role values."""
    with pytest.raises(ValidationError):
        ChatMessage(role=cast(Any, "invalid"), content="hello")


def test_chat_completion_request_minimal() -> None:
    """ChatCompletionRequest should work with minimal fields."""
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hello")]
    )
    assert request.model is None
    assert request.stream is False
    assert request.n == 1


def test_chat_completion_request_all_fields() -> None:
    """ChatCompletionRequest should accept all fields."""
    request = ChatCompletionRequest(
        model="gpt-4",
        messages=[ChatMessage(role="user", content="hello")],
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
        user="user-123",
    )
    assert request.model == "gpt-4"
    assert request.temperature == 0.7
    assert request.top_p == 0.9
    assert request.max_tokens == 100
    assert request.user == "user-123"


def test_chat_completion_request_extra_fields_allowed() -> None:
    """ChatCompletionRequest should allow extra fields."""
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [ChatMessage(role="user", content="hello")],
            "frequency_penalty": 0.5,
        }
    )
    assert hasattr(request, "frequency_penalty")


def test_chat_completion_request_chat_kwargs() -> None:
    """chat_kwargs should format correctly."""
    request = ChatCompletionRequest(
        messages=[
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="hello"),
        ],
        temperature=0.5,
        top_p=0.8,
        max_tokens=50,
        user="user-1",
    )

    kwargs = request.chat_kwargs()

    assert kwargs["messages"] == [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "hello"},
    ]
    assert kwargs["temperature"] == 0.5
    assert kwargs["top_p"] == 0.8
    assert kwargs["max_tokens"] == 50
    assert kwargs["user"] == "user-1"


def test_chat_completion_request_chat_kwargs_excludes_none() -> None:
    """chat_kwargs should exclude None values."""
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hello")],
    )

    kwargs = request.chat_kwargs()

    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert "max_tokens" not in kwargs
    assert "user" not in kwargs


def test_embeddings_request_single_input() -> None:
    """EmbeddingsRequest should accept single string input."""
    request = EmbeddingsRequest(input="hello world")
    assert request.input == "hello world"


def test_embeddings_request_list_input() -> None:
    """EmbeddingsRequest should accept list input."""
    request = EmbeddingsRequest(input=["hello", "world"])
    assert request.input == ["hello", "world"]


def test_embeddings_request_all_fields() -> None:
    """EmbeddingsRequest should accept all fields."""
    request = EmbeddingsRequest(
        model="embedding-model",
        input=["text1", "text2"],
        encoding_format="float",
        dimensions=1536,
        user="user-1",
    )
    assert request.model == "embedding-model"
    assert request.dimensions == 1536
    assert request.encoding_format == "float"
    assert request.user == "user-1"


def test_json_schema_model_factory_simple_object() -> None:
    """JsonSchemaModelFactory should build model from simple object schema."""
    definition = ResponseFormatJsonSchemaDefinition(
        name="TestModel",
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        },
    )

    model = JsonSchemaModelFactory.build_model(definition)

    assert issubclass(model, BaseModel)
    instance = model(name="Alice", age=30)
    dumped = instance.model_dump()
    assert dumped["name"] == "Alice"
    assert dumped["age"] == 30


def test_json_schema_model_factory_optional_fields() -> None:
    """JsonSchemaModelFactory should handle optional fields."""
    definition = ResponseFormatJsonSchemaDefinition(
        name="TestModel",
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
            "required": ["name"],
        },
    )

    model = JsonSchemaModelFactory.build_model(definition)

    instance = model(name="Alice")
    dumped = instance.model_dump()
    assert dumped["name"] == "Alice"
    assert dumped["email"] is None


def test_json_schema_model_factory_all_types() -> None:
    """JsonSchemaModelFactory should handle all JSON schema types."""
    definition = ResponseFormatJsonSchemaDefinition(
        name="AllTypes",
        schema={
            "type": "object",
            "properties": {
                "string_field": {"type": "string"},
                "int_field": {"type": "integer"},
                "float_field": {"type": "number"},
                "bool_field": {"type": "boolean"},
                "array_field": {"type": "array"},
                "object_field": {"type": "object"},
            },
            "required": [
                "string_field",
                "int_field",
                "float_field",
                "bool_field",
                "array_field",
                "object_field",
            ],
        },
    )

    model = JsonSchemaModelFactory.build_model(definition)

    instance = model(
        string_field="text",
        int_field=42,
        float_field=3.14,
        bool_field=True,
        array_field=[1, 2, 3],
        object_field={"key": "value"},
    )

    dumped = instance.model_dump()
    assert dumped["string_field"] == "text"
    assert dumped["int_field"] == 42
    assert dumped["float_field"] == 3.14
    assert dumped["bool_field"] is True
    assert dumped["array_field"] == [1, 2, 3]
    assert dumped["object_field"] == {"key": "value"}


def test_json_schema_model_factory_invalid_root_type() -> None:
    """JsonSchemaModelFactory should reject non-object root types."""
    definition = ResponseFormatJsonSchemaDefinition(
        name="Invalid",
        schema={"type": "string"},
    )

    with pytest.raises(ValueError) as exc_info:
        JsonSchemaModelFactory.build_model(definition)

    assert "root object" in str(exc_info.value).lower()


def test_json_schema_model_factory_invalid_properties_not_dict() -> None:
    """JsonSchemaModelFactory should validate properties is a dict."""
    definition = ResponseFormatJsonSchemaDefinition(
        name="Invalid",
        schema={
            "type": "object",
            "properties": ["not", "a", "dict"],
        },
    )

    with pytest.raises(ValueError) as exc_info:
        JsonSchemaModelFactory.build_model(definition)

    assert "properties must be an object" in str(exc_info.value).lower()


def test_json_schema_model_factory_invalid_property_schema_not_dict() -> None:
    """JsonSchemaModelFactory should validate each property schema is a dict."""
    definition = ResponseFormatJsonSchemaDefinition(
        name="Invalid",
        schema={
            "type": "object",
            "properties": {
                "field": "invalid",
            },
        },
    )

    with pytest.raises(ValueError) as exc_info:
        JsonSchemaModelFactory.build_model(definition)

    assert "property schema must be an object" in str(exc_info.value).lower()


def test_response_format_json_schema_definition_with_alias() -> None:
    """ResponseFormatJsonSchemaDefinition should map schema -> schema_."""
    definition = ResponseFormatJsonSchemaDefinition(
        name="Test",
        schema={"type": "object", "properties": {}},
    )
    assert definition.schema_ == {"type": "object", "properties": {}}


def test_response_format_json_schema() -> None:
    """ResponseFormatJsonSchema should validate type."""
    fmt = ResponseFormatJsonSchema(
        type="json_schema",
        json_schema=ResponseFormatJsonSchemaDefinition(
            name="Test",
            schema={"type": "object", "properties": {}},
        ),
    )
    assert fmt.type == "json_schema"


def test_chat_completion_usage_defaults() -> None:
    """ChatCompletionUsage should have zero defaults."""
    from llm_shim.api.models import ChatCompletionUsage

    usage = ChatCompletionUsage()
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


def test_chat_completion_response() -> None:
    """ChatCompletionResponse should build correctly."""
    from llm_shim.api.models import (
        ChatCompletionChoice,
        ChatCompletionResponse,
        ChatCompletionResponseMessage,
        ChatCompletionUsage,
    )

    response = ChatCompletionResponse(
        id="chatcmpl-123",
        created=1234567890,
        model="gpt-4",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionResponseMessage(content="response text"),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(),
    )

    assert response.object == "chat.completion"
    assert response.id == "chatcmpl-123"
    assert response.model == "gpt-4"
    assert len(response.choices) == 1


def test_embeddings_response() -> None:
    """EmbeddingsResponse should build correctly."""
    from llm_shim.api.models import EmbeddingDatum, EmbeddingsResponse, EmbeddingsUsage

    response = EmbeddingsResponse(
        data=[
            EmbeddingDatum(index=0, embedding=[1.0, 2.0, 3.0]),
            EmbeddingDatum(index=1, embedding=[4.0, 5.0, 6.0]),
        ],
        model="embedding-model",
        usage=EmbeddingsUsage(),
    )

    assert response.object == "list"
    assert response.model == "embedding-model"
    assert len(response.data) == 2
    assert response.data[0].object == "embedding"


def test_json_schema_model_factory_unknown_type() -> None:
    """_annotation_from_schema should fallback to dict for unknown types."""
    definition = ResponseFormatJsonSchemaDefinition(
        name="UnknownTypes",
        schema={
            "type": "object",
            "properties": {
                "unknown": {"type": "null"},
                "also_unknown": {"type": "enum"},
            },
            "required": ["unknown", "also_unknown"],
        },
    )

    model = JsonSchemaModelFactory.build_model(definition)

    # Unknown types should be treated as dict[str, Any]
    instance = model(
        unknown={"anything": "goes"},
        also_unknown={"options": ["a", "b"]},
    )
    dumped = instance.model_dump()
    assert dumped["unknown"] == {"anything": "goes"}
    assert dumped["also_unknown"] == {"options": ["a", "b"]}
