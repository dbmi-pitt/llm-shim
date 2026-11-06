"""Request-level limits applied after authentication and before provider routing."""

from llm_shim.api.schemas.openai import ChatCompletionRequest, EmbeddingsRequest
from llm_shim.core.config import SecuritySettings
from llm_shim.core.exceptions import BadRequestError


def validate_chat_request(
    request: ChatCompletionRequest, security: SecuritySettings
) -> None:
    """Apply configured message and output-budget bounds."""
    if len(request.messages) > security.max_messages:
        raise BadRequestError("Too many messages")
    total = 0
    for message in request.messages:
        if len(message.content) > security.max_message_chars:
            raise BadRequestError("A message is too large")
        total += len(message.content)
    if total > security.max_total_input_chars:
        raise BadRequestError("Total input is too large")
    if request.max_tokens is not None and request.max_tokens > security.max_tokens:
        raise BadRequestError("max_tokens exceeds the configured limit")


def validate_embeddings_request(
    request: EmbeddingsRequest, security: SecuritySettings
) -> None:
    """Apply configured embedding batch and input-size bounds."""
    values = [request.input] if isinstance(request.input, str) else request.input
    if not values or len(values) > security.max_embedding_inputs:
        raise BadRequestError("Embedding input batch is invalid or too large")
    if any(len(value) > security.max_embedding_chars for value in values):
        raise BadRequestError("An embedding input is too large")
    if sum(len(value) for value in values) > security.max_total_input_chars:
        raise BadRequestError("Total input is too large")
