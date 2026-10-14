from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AppBaseException
from app.core.logging import logger
from app.db.session import get_db
from app.schemas.chat import (
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
)
from app.services.memory import chat_memory
from app.services.rag import rag_pipeline

router = APIRouter(prefix="/chat", tags=["Conversational RAG & Booking"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Multi-turn Conversational RAG with Redis Memory and Interview Booking",
    description=(
        "Processes a user message within a multi-turn conversation session. "
        "Leverages Redis for chat memory context, performs custom RAG over Qdrant without RetrievalQAChain, "
        "and detects interview scheduling intent to automatically extract (name, email, date, time) "
        "and record the booking in the relational database."
    ),
)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    try:
        response = await rag_pipeline.execute(request=request, db=db)
        return response
    except AppBaseException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during conversational RAG: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred while processing your request.",
        )


@router.get(
    "/{session_id}/history",
    response_model=ChatHistoryResponse,
    summary="Retrieve session chat history stored in Redis",
)
async def get_session_history(
    session_id: str,
    limit: int = 50,
) -> ChatHistoryResponse:
    messages = await chat_memory.get_messages(session_id=session_id, limit=limit)
    return ChatHistoryResponse(
        session_id=session_id,
        total_messages=len(messages),
        messages=messages,
    )


@router.delete(
    "/{session_id}/history",
    status_code=status.HTTP_200_OK,
    summary="Clear conversation memory from Redis for a session",
)
async def clear_session_history(
    session_id: str,
) -> dict:
    await chat_memory.clear_history(session_id=session_id)
    return {
        "session_id": session_id,
        "message": "Chat history cleared successfully.",
    }
