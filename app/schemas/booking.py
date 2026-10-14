from datetime import datetime, timezone
import re
from typing import Optional, List
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class BookingCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=150, description="Full name of interviewee")
    email: EmailStr = Field(..., description="Valid contact email address")
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Interview date in YYYY-MM-DD format")
    time: str = Field(..., min_length=2, max_length=20, description="Interview time (e.g. 14:00 or 2:00 PM)")
    session_id: Optional[str] = Field(default=None, description="Optional chat session ID")
    notes: Optional[str] = Field(default=None, description="Additional candidate notes or interview topic")

    @field_validator("date")
    @classmethod
    def validate_future_date(cls, v: str) -> str:
        """Validates that the booking date is today or in the future."""
        try:
            parsed_date = datetime.strptime(v, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError("Date must be in valid YYYY-MM-DD format.")

        today = datetime.now(timezone.utc).date()
        if parsed_date < today:
            raise ValueError(f"Booking date '{v}' cannot be in the past. Please choose a future date.")
        return v

    @field_validator("time")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        """Validates that time format follows common patterns like 14:00 or 2:00 PM."""
        cleaned = v.strip()
        pattern = r"^(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)|\d{1,2}:\d{2})$"
        if not re.match(pattern, cleaned):
            raise ValueError(
                f"Invalid time format '{v}'. Please provide a valid time such as '14:00' or '2:00 PM'."
            )
        return cleaned


class BookingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: Optional[str] = None
    name: str
    email: str
    date: str
    time: str
    status: str
    notes: Optional[str] = None
    created_at: datetime


class BookingIntentExtraction(BaseModel):
    """Schema returned by LLM function calling / structured output for booking extraction."""
    is_booking_intent: bool = Field(
        ...,
        description="Whether the user expressed intent to schedule, book, or reserve an interview",
    )
    name: Optional[str] = Field(default=None, description="Candidate name if provided")
    email: Optional[str] = Field(default=None, description="Candidate email if provided")
    date: Optional[str] = Field(default=None, description="Interview date (YYYY-MM-DD) if provided")
    time: Optional[str] = Field(default=None, description="Interview time (e.g., 14:00 or 10:00 AM) if provided")
    missing_fields: List[str] = Field(
        default_factory=list,
        description="List of required fields still missing among name, email, date, time",
    )
