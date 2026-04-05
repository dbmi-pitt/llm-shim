"""Chat completion API endpoint."""

from fastapi import APIRouter, Depends

from llm_shim.api.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from llm_shim.services.chat import ChatService, get_chat_service

__all__ = ["router"]

router = APIRouter(tags=["chat"])


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatCompletionResponse:
    """Create an OpenAI-compatible chat completion response.

    Args:
        request (ChatCompletionRequest): OpenAI chat completion request.
        service (ChatService): Injected chat service.

    Returns:
        ChatCompletionResponse: OpenAI chat completion response.
    """
    return await service.create(request)
