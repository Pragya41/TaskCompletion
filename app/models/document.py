import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import String, Integer, Text, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    """Stores ingested document metadata and ingestion parameters."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    chunking_strategy: Mapped[str] = mapped_column(String(50), nullable=False)
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_overlap: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    vector_collection: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="PROCESSED", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class DocumentChunk(Base):
    """Stores metadata and snippet of individual text chunks for an ingested document."""

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    vector_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
