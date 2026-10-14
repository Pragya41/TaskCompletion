import pytest
from app.schemas.chat import ChatRequest
from app.services.rag import rag_pipeline
from app.services.memory import chat_memory


@pytest.mark.asyncio
async def test_rag_pipeline_general_query(db_session):
    session_id = "test-rag-session-1"
    await chat_memory.clear_history(session_id)

    request = ChatRequest(
        session_id=session_id,
        query="What services does Palm Mind AI offer?",
    )
    response = await rag_pipeline.execute(request, db_session)

    assert response.session_id == session_id
    assert response.response is not None
    assert response.booking is None

    # Check that Redis recorded the interaction
    history = await chat_memory.get_messages(session_id)
    assert len(history) == 2  # user + assistant


@pytest.mark.asyncio
async def test_rag_pipeline_booking_flow(db_session):
    session_id = "test-rag-booking-flow"
    await chat_memory.clear_history(session_id)

    # 1. Booking query with full info
    request = ChatRequest(
        session_id=session_id,
        query="I want to book an interview. My name is Alex Carter, email alex@example.com for date 2026-10-28 at 10:00 AM.",
    )
    response = await rag_pipeline.execute(request, db_session)

    assert response.booking is not None
    assert response.booking.name == "Alex Carter"
    assert response.booking.email == "alex@example.com"
    assert "scheduled successfully" in response.response.lower()
