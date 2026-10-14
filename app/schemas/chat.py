from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.booking import BookingResponse


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    role: ChatRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100, description="Unique conversation session identifier")
    query: str = Field(..., min_length=1, description="User question, query, or booking statement")
    document_id: Optional[str] = Field(
        default=None,
        description="Optional document ID to restrict retrieval context to a specific uploaded document",
    )
    top_k: int = Field(default=4, ge=1, le=20, description="Number of context chunks to retrieve from vector store")


class RetrievedSource(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    content: str
    score: float


class ChatResponse(BaseModel):
    session_id: str
    query: str
    rewritten_query: Optional[str] = None
    response: str
    sources: List[RetrievedSource] = []
    booking: Optional[BookingResponse] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatHistoryResponse(BaseModel):
    session_id: str
    total_messages: int
    messages: List[ChatMessage]
