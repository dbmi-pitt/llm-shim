"""Pydantic models for OpenAI-compatible request and response payloads."""

from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, create_model


class ChatMessage(BaseModel):
    """OpenAI-style chat message payload."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class ResponseFormatText(BaseModel):
    """OpenAI response format for plain text."""

    type: Literal["text"]


class ResponseFormatJsonSchemaDefinition(BaseModel):
    """OpenAI json_schema response format definition."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = "response"
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool | None = None


class ResponseFormatJsonSchema(BaseModel):
    """OpenAI response format for schema-validated JSON."""

    type: Literal["json_schema"]
    json_schema: ResponseFormatJsonSchemaDefinition


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    n: int = 1
    stream: Literal[False] = False
    response_format: ResponseFormatText | ResponseFormatJsonSchema | None = None
    user: str | None = None

    def chat_kwargs(self) -> dict[str, Any]:
        """Build kwargs forwarded to the pydantic-ai chat model."""
        kwargs: dict[str, Any] = {
            "messages": [
                message.model_dump(exclude_none=True) for message in self.messages
            ],
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.user is not None:
            kwargs["user"] = self.user
        return kwargs


class ChatCompletionUsage(BaseModel):
    """OpenAI-compatible token usage details."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponseMessage(BaseModel):
    """Assistant message in completion choices."""

    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    """A chat completion choice entry."""

    index: int = 0
    message: ChatCompletionResponseMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class EmbeddingsRequest(BaseModel):
    """OpenAI-compatible embeddings request."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: str | list[str]
    encoding_format: Literal["float"] | None = None
    dimensions: int | None = None
    user: str | None = None


class EmbeddingDatum(BaseModel):
    """OpenAI-compatible single embedding entry."""

    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class EmbeddingsUsage(BaseModel):
    """OpenAI-compatible embeddings usage section."""

    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingsResponse(BaseModel):
    """OpenAI-compatible embeddings response."""

    object: Literal["list"] = "list"
    data: list[EmbeddingDatum]
    model: str
    usage: EmbeddingsUsage


class JsonSchemaModelFactory:
    """Build dynamic Pydantic models from a subset of JSON Schema."""

    @classmethod
    def build_model(
        cls,
        definition: ResponseFormatJsonSchemaDefinition,
    ) -> type[BaseModel]:
        """Create a dynamic model for OpenAI `response_format.json_schema`."""
        return cls._build_object_model(definition.name, definition.schema_)

    @classmethod
    def _build_object_model(
        cls,
        name: str,
        schema: dict[str, Any],
    ) -> type[BaseModel]:
        """Build a model from an object schema with `properties` and `required`."""
        schema_type = schema.get("type")
        if schema_type != "object":
            msg = (
                "Only root object schemas are supported for response_format.json_schema"
            )
            raise ValueError(msg)

        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            msg = "Schema properties must be an object"
            raise ValueError(msg)

        required = schema.get("required", [])
        required_fields = set(required) if isinstance(required, list) else set()

        model_fields: dict[str, tuple[Any, Any]] = {}
        for field_name, field_schema in properties.items():
            if not isinstance(field_schema, dict):
                msg = "Each property schema must be an object"
                raise ValueError(msg)
            annotation = cls._annotation_from_schema(
                name=f"{name}_{field_name}",
                schema=field_schema,
            )
            if field_name in required_fields:
                model_fields[field_name] = (annotation, ...)
            else:
                optional_annotation: Any = annotation | None
                model_fields[field_name] = (optional_annotation, None)

        return cast(
            type[BaseModel],
            create_model(name, **model_fields),  # ty: ignore[no-matching-overload]
        )

    @classmethod
    def _annotation_from_schema(cls, name: str, schema: dict[str, Any]) -> type[Any]:
        """Map JSON Schema nodes into Python/Pydantic annotations."""
        schema_type = schema.get("type")

        if schema_type == "string":
            return str
        if schema_type == "integer":
            return int
        if schema_type == "number":
            return float
        if schema_type == "boolean":
            return bool
        if schema_type == "array":
            return list[Any]
        if schema_type == "object":
            return dict[str, Any]

        return dict[str, Any]
