import pytest
from app.schemas.chat import ChatRole
from app.services.memory import chat_memory


@pytest.mark.asyncio
async def test_redis_chat_memory_lifecycle():
    session_id = "test-session-123"

    # Clean prior state if any
    await chat_memory.clear_history(session_id)

    # 1. Add user message
    msg1 = await chat_memory.add_message(session_id, ChatRole.USER, "Hello Palm Mind AI!")
    assert msg1.role == ChatRole.USER
    assert msg1.content == "Hello Palm Mind AI!"

    # 2. Add assistant message
    msg2 = await chat_memory.add_message(session_id, ChatRole.ASSISTANT, "Greetings! How can I help you today?")
    assert msg2.role == ChatRole.ASSISTANT

    # 3. Retrieve messages
    messages = await chat_memory.get_messages(session_id)
    assert len(messages) == 2
    assert messages[0].content == "Hello Palm Mind AI!"
    assert messages[1].content == "Greetings! How can I help you today?"

    # 4. Formatted history
    formatted = await chat_memory.get_formatted_history(session_id)
    assert "User: Hello Palm Mind AI!" in formatted
    assert "Assistant: Greetings! How can I help you today?" in formatted

    # 5. Session state flag testing (e.g. BOOKING_IN_PROGRESS)
    assert await chat_memory.get_session_state(session_id) is None
    await chat_memory.set_session_state(session_id, "BOOKING_IN_PROGRESS")
    assert await chat_memory.get_session_state(session_id) == "BOOKING_IN_PROGRESS"
    await chat_memory.clear_session_state(session_id)
    assert await chat_memory.get_session_state(session_id) is None

    # 6. Clear history
    cleared = await chat_memory.clear_history(session_id)
    assert cleared is True

    messages_after = await chat_memory.get_messages(session_id)
    assert len(messages_after) == 0
