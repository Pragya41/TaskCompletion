import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Booking(Base):
    """Stores interview bookings generated via Conversational RAG agent."""

    __tablename__ = "bookings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    date: Mapped[str] = mapped_column(String(20), nullable=False)  # e.g., '2026-10-20'
    time: Mapped[str] = mapped_column(String(20), nullable=False)  # e.g., '14:30' or '2:30 PM'
    status: Mapped[str] = mapped_column(String(30), default="CONFIRMED", nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
