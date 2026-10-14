import pytest
from pydantic import ValidationError
from app.schemas.booking import BookingCreate
from app.services.booking.booking_service import booking_service
from app.services.llm.llm_service import llm_service


@pytest.mark.asyncio
async def test_create_and_fetch_booking(db_session):
    booking_in = BookingCreate(
        name="Pragya Dhital",
        email="pragya@example.com",
        date="2026-10-25",
        time="14:00",
        notes="Hiring Task Interview",
    )
    booking = await booking_service.create_booking(db_session, booking_in)

    assert booking.id is not None
    assert booking.name == "Pragya Dhital"
    assert booking.email == "pragya@example.com"
    assert booking.status == "CONFIRMED"

    # Fetch by ID
    fetched = await booking_service.get_booking_by_id(db_session, booking.id)
    assert fetched is not None
    assert fetched.id == booking.id

    # List bookings
    all_bookings = await booking_service.list_bookings(db_session)
    assert len(all_bookings) >= 1


def test_booking_validation_past_date():
    """Verify that dates in the past are rejected."""
    with pytest.raises(ValidationError) as exc:
        BookingCreate(
            name="John Doe",
            email="john@example.com",
            date="2020-01-01",
            time="14:00",
        )
    assert "cannot be in the past" in str(exc.value)


def test_booking_validation_invalid_time():
    """Verify that invalid time formats are rejected."""
    with pytest.raises(ValidationError) as exc:
        BookingCreate(
            name="John Doe",
            email="john@example.com",
            date="2026-10-25",
            time="noonish",
        )
    assert "Invalid time format" in str(exc.value)


@pytest.mark.asyncio
async def test_llm_booking_intent_extraction_full():
    query = "Please schedule an interview for candidate Pragya Dhital, email pragya@example.com on 2026-10-25 at 2:00 PM."
    extracted = await llm_service.extract_booking_intent(query)

    assert extracted.is_booking_intent is True
    assert extracted.name == "Pragya Dhital"
    assert extracted.email == "pragya@example.com"
    assert extracted.date == "2026-10-25"
    assert extracted.time is not None
    assert len(extracted.missing_fields) == 0


@pytest.mark.asyncio
async def test_llm_booking_intent_extraction_partial():
    query = "I would like to schedule an interview please."
    extracted = await llm_service.extract_booking_intent(query)

    assert extracted.is_booking_intent is True
    assert len(extracted.missing_fields) > 0


@pytest.mark.asyncio
async def test_no_false_booking_intent_on_old_history():
    """Test Bug 4 fix: Once booking is done, normal questions are NOT treated as bookings."""
    old_history = (
        "User: I want to book an interview for Pragya Dhital, pragya@example.com on 2026-10-25 at 14:00\n"
        "Assistant: Your interview has been scheduled successfully!\n• Candidate: Pragya Dhital"
    )
    normal_query = "What is Palm Mind AI's core value?"

    extracted = await llm_service.extract_booking_intent(
        query=normal_query,
        is_booking_in_progress=False,
        chat_history=old_history,
    )
    # Must NOT be treated as a booking intent!
    assert extracted.is_booking_intent is False
