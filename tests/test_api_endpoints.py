import io
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert data["redis"] == "connected"
    assert data["qdrant"] == "connected"


@pytest.mark.asyncio
async def test_root(client: AsyncClient):
    resp = await client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert "/docs" in data["documentation"]


@pytest.mark.asyncio
async def test_ingest_txt_file_recursive(client: AsyncClient):
    content = (
        "Palm Mind AI develops scalable AI architectures.\n\n"
        "Our Conversational RAG engine uses Redis for persistent memory.\n\n"
        "Technical interviews can be booked through automated agents."
    ).encode("utf-8")

    files = {"file": ("palm_overview.txt", content, "text/plain")}
    data = {
        "chunking_strategy": "recursive_character",
        "chunk_size": "100",
        "chunk_overlap": "20",
    }

    resp = await client.post("/api/v1/documents/ingest", files=files, data=data)
    assert resp.status_code == 201
    body = resp.json()
    assert body["filename"] == "palm_overview.txt"
    assert body["chunking_strategy"] == "recursive_character"
    assert body["total_chunks"] >= 3
    assert len(body["sample_chunks"]) > 0

    doc_id = body["document_id"]

    # Verify listing
    list_resp = await client.get("/api/v1/documents")
    assert list_resp.status_code == 200
    docs = list_resp.json()
    assert any(d["id"] == doc_id for d in docs)

    # Verify single document detail
    detail_resp = await client.get(f"/api/v1/documents/{doc_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["id"] == doc_id


@pytest.mark.asyncio
async def test_ingest_pdf_file_fixed(client: AsyncClient):
    with open("sample_data/palm_mind_handbook.pdf", "rb") as f:
        pdf_bytes = f.read()

    files = {"file": ("palm_mind_handbook.pdf", pdf_bytes, "application/pdf")}
    data = {
        "chunking_strategy": "fixed_size",
        "chunk_size": "200",
        "chunk_overlap": "30",
    }

    resp = await client.post("/api/v1/documents/ingest", files=files, data=data)
    assert resp.status_code == 201
    body = resp.json()
    assert body["filename"] == "palm_mind_handbook.pdf"
    assert body["file_type"] == "pdf"
    assert body["chunking_strategy"] == "fixed_size"
    assert body["total_chunks"] > 0


@pytest.mark.asyncio
async def test_ingest_unsupported_file(client: AsyncClient):
    files = {"file": ("test.png", b"fake binary", "image/png")}
    resp = await client.post("/api/v1/documents/ingest", files=files)
    assert resp.status_code == 415


@pytest.mark.asyncio
async def test_conversational_chat_and_memory(client: AsyncClient):
    session_id = "integration-chat-session"

    # Turn 1
    resp1 = await client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "query": "What is Palm Mind AI?"},
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["session_id"] == session_id
    assert data1["response"] is not None

    # Turn 2: Check history
    hist_resp = await client.get(f"/api/v1/chat/{session_id}/history")
    assert hist_resp.status_code == 200
    assert hist_resp.json()["total_messages"] >= 2

    # Clear history
    del_resp = await client.delete(f"/api/v1/chat/{session_id}/history")
    assert del_resp.status_code == 200

    hist_after = await client.get(f"/api/v1/chat/{session_id}/history")
    assert hist_after.json()["total_messages"] == 0


@pytest.mark.asyncio
async def test_conversational_interview_booking_endpoint(client: AsyncClient):
    session_id = "integration-booking-session"

    payload = {
        "session_id": session_id,
        "query": "I would like to schedule an interview. My name is Maya Lin, email maya@example.com for 2026-10-30 at 15:00.",
    }
    resp = await client.post("/api/v1/chat", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["booking"] is not None
    assert body["booking"]["name"] == "Maya Lin"
    assert body["booking"]["email"] == "maya@example.com"
    booking_id = body["booking"]["id"]

    # Verify booking retrieval via /api/v1/bookings
    b_resp = await client.get(f"/api/v1/bookings/{booking_id}")
    assert b_resp.status_code == 200
    assert b_resp.json()["name"] == "Maya Lin"

    # Turn 3: Ask a normal question after booking completed to ensure no false booking trigger!
    resp_followup = await client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "query": "What are your company core values?"},
    )
    assert resp_followup.status_code == 200
    assert resp_followup.json()["booking"] is None


@pytest.mark.asyncio
async def test_conversational_booking_invalid_date_graceful_reply(client: AsyncClient):
    """Test that submitting a past date in chat replies with a helpful conversational error instead of 500."""
    session_id = "invalid-date-session"

    payload = {
        "session_id": session_id,
        "query": "I want to schedule an interview for John, email john@example.com on 2020-01-01 at 14:00.",
    }
    resp = await client.post("/api/v1/chat", json=payload)
    # Must NOT return 500 error!
    assert resp.status_code == 200
    data = resp.json()
    assert data["booking"] is None
    # Politeness and clarity check
    assert "cannot be in the past" in data["response"] or "future date" in data["response"].lower()
