from typing import Any, Dict, Optional
from fastapi import HTTPException, status


class AppBaseException(Exception):
    """Base exception for application domain errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DocumentExtractionError(AppBaseException):
    """Raised when text extraction from a file fails."""
    pass


class InvalidFileTypeError(AppBaseException):
    """Raised when an unsupported file type is uploaded."""
    pass


class ChunkingError(AppBaseException):
    """Raised when document chunking fails."""
    pass


class VectorStoreError(AppBaseException):
    """Raised when an operation against the vector database fails."""
    pass


class RedisMemoryError(AppBaseException):
    """Raised when conversational memory retrieval or persistence fails."""
    pass


class BookingError(AppBaseException):
    """Raised when interview booking logic fails."""
    pass


class LLMServiceError(AppBaseException):
    """Raised when LLM invocation or parsing fails."""
    pass
