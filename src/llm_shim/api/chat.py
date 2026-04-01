"""Chat completion API endpoint."""

from fastapi import APIRouter, HTTPException

from llm_shim.api.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from llm_shim.services.chat import ChatService

__all__ = ["router"]

router = APIRouter(tags=["chat"])


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Create an OpenAI-compatible chat completion response.

    Args:
        request (ChatCompletionRequest): OpenAI chat completion request.
        service (ChatService): Injected chat service.

    Returns:
        ChatCompletionResponse: OpenAI chat completion response.
    """
    service = ChatService()
    try:
        return await service.create(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        detail = str(error)
        if detail.startswith("Failed to initialize provider client:"):
            raise HTTPException(status_code=500, detail=detail) from error
        raise HTTPException(status_code=502, detail=detail) from error
