"""Chat completion API endpoint."""

from fastapi import APIRouter, Depends, Request

from llm_shim.api.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from llm_shim.api.validation import validate_chat_request
from llm_shim.services.chat import ChatService, get_chat_service

__all__ = ["router"]

router = APIRouter(tags=["chat"])


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    http_request: Request,
    service: ChatService = Depends(get_chat_service),
) -> ChatCompletionResponse:
    """Create an OpenAI-compatible chat completion response.

    Args:
        request (ChatCompletionRequest): OpenAI chat completion request.
        service (ChatService): Injected chat service.
        http_request (Request): HTTP request used for configured limits.

    Returns:
        ChatCompletionResponse: OpenAI chat completion response.
    """
    validate_chat_request(request, http_request.app.state.settings.security)
    return await service.create(request)
