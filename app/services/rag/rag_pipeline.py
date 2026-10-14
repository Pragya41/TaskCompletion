from typing import Optional
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.models.booking import Booking
from app.schemas.booking import BookingCreate, BookingResponse
from app.schemas.chat import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatRole,
    RetrievedSource,
)
from app.services.booking.booking_service import booking_service
from app.services.embeddings.embedding_service import embedding_service
from app.services.llm.llm_service import llm_service
from app.services.memory.redis_memory import chat_memory
from app.services.vector_store.qdrant_store import vector_store


class ConversationalRAGPipeline:
    """Custom RAG pipeline handling chat memory, document search, and interview booking."""

    async def execute(
        self,
        request: ChatRequest,
        db: AsyncSession,
    ) -> ChatResponse:
        session_id = request.session_id
        raw_query = request.query.strip()

        # 1. Check if a booking is currently in progress via Redis state flag
        session_state = await chat_memory.get_session_state(session_id)
        is_booking_in_progress = (session_state == "BOOKING_IN_PROGRESS")

        # 2. Get recent chat history from Redis
        chat_history = await chat_memory.get_formatted_history(session_id, limit=6)

        # 3. Save the incoming user message into Redis
        await chat_memory.add_message(
            session_id=session_id,
            role=ChatRole.USER,
            content=raw_query,
        )

        # 4. Check for interview booking intent
        # If not in progress, only the latest query triggers booking intent (prevents re-triggering on old history)
        booking_intent = await llm_service.extract_booking_intent(
            query=raw_query,
            is_booking_in_progress=is_booking_in_progress,
            chat_history=chat_history,
        )

        if booking_intent.is_booking_intent:
            # Case A: All required details (name, email, date, time) are present
            if not booking_intent.missing_fields and booking_intent.name and booking_intent.email and booking_intent.date and booking_intent.time:
                # Validate details using BookingCreate schema (checks future date & valid time)
                try:
                    booking_in = BookingCreate(
                        name=booking_intent.name,
                        email=booking_intent.email,
                        date=booking_intent.date,
                        time=booking_intent.time,
                        session_id=session_id,
                        notes=f"Scheduled via Conversational RAG agent. Query: {raw_query}",
                    )
                except ValidationError as val_err:
                    # Catch date/time validation errors and reply with a conversational error message
                    first_err = val_err.errors()[0]
                    raw_msg = first_err.get("msg", "Please provide a valid future date and time.")
                    clean_msg = raw_msg.replace("Value error, ", "").strip()

                    validation_reply = (
                        f"I noticed an issue with the booking details: {clean_msg} "
                        f"Could you please provide an updated date or time?"
                    )

                    # Keep booking state in progress in Redis so user can reply with the corrected value
                    await chat_memory.set_session_state(session_id, "BOOKING_IN_PROGRESS")
                    await chat_memory.add_message(
                        session_id=session_id,
                        role=ChatRole.ASSISTANT,
                        content=validation_reply,
                    )

                    return ChatResponse(
                        session_id=session_id,
                        query=raw_query,
                        rewritten_query=raw_query,
                        response=validation_reply,
                        sources=[],
                        booking=None,
                    )

                # Save the booking in the SQL database
                logger.info(f"Booking interview for candidate {booking_in.name} (session: {session_id})")
                booking_record = await booking_service.create_booking(db=db, booking_in=booking_in)

                # Clear in-progress flag in Redis now that booking is complete
                await chat_memory.clear_session_state(session_id)

                # Prepare confirmation message for candidate
                confirmation_msg = (
                    f"Your interview has been scheduled successfully!\n\n"
                    f"• Candidate: {booking_record.name}\n"
                    f"• Email: {booking_record.email}\n"
                    f"• Date: {booking_record.date}\n"
                    f"• Time: {booking_record.time}\n"
                    f"• Status: {booking_record.status}\n"
                    f"• Booking Reference ID: {booking_record.id}\n\n"
                    f"A confirmation has been recorded. Please let me know if you need to reschedule or have any questions about the position."
                )

                # Save the assistant response in Redis
                await chat_memory.add_message(
                    session_id=session_id,
                    role=ChatRole.ASSISTANT,
                    content=confirmation_msg,
                )

                return ChatResponse(
                    session_id=session_id,
                    query=raw_query,
                    rewritten_query=raw_query,
                    response=confirmation_msg,
                    sources=[],
                    booking=BookingResponse.model_validate(booking_record),
                )
            else:
                # Case B: Some details are missing; mark state as in-progress and prompt user
                await chat_memory.set_session_state(session_id, "BOOKING_IN_PROGRESS")

                missing_str = ", ".join(booking_intent.missing_fields)
                clarification_msg = (
                    f"I would be glad to help schedule your interview! To confirm the reservation, "
                    f"could you please provide the following missing details: {missing_str}?"
                )

                # Save assistant clarification request in Redis
                await chat_memory.add_message(
                    session_id=session_id,
                    role=ChatRole.ASSISTANT,
                    content=clarification_msg,
                )

                return ChatResponse(
                    session_id=session_id,
                    query=raw_query,
                    rewritten_query=raw_query,
                    response=clarification_msg,
                    sources=[],
                    booking=None,
                )

        # 5. If not a booking, ensure booking state is cleared
        if is_booking_in_progress:
            await chat_memory.clear_session_state(session_id)

        # 6. Rewrite the query using conversation history for better search
        rewritten_query = await llm_service.rewrite_query(
            query=raw_query,
            chat_history=chat_history,
        )

        # 7. Convert query to vector embedding and search Qdrant for matching chunks
        query_vector = await embedding_service.embed_query(rewritten_query)
        retrieved_hits = await vector_store.search(
            query_vector=query_vector,
            top_k=request.top_k,
            document_id=request.document_id,
        )

        # 8. Build sources list for citations
        sources = [
            RetrievedSource(
                document_id=hit["document_id"],
                filename=hit["filename"],
                chunk_index=hit["chunk_index"],
                content=hit["text"],
                score=round(hit["score"], 4),
            )
            for hit in retrieved_hits
        ]

        # 9. Generate grounded answer with LLM using the retrieved chunks
        answer = await llm_service.generate_rag_answer(
            query=rewritten_query,
            context_chunks=retrieved_hits,
            chat_history=chat_history,
        )

        # 10. Save the final assistant answer into Redis
        await chat_memory.add_message(
            session_id=session_id,
            role=ChatRole.ASSISTANT,
            content=answer,
        )

        return ChatResponse(
            session_id=session_id,
            query=raw_query,
            rewritten_query=rewritten_query,
            response=answer,
            sources=sources,
            booking=None,
        )


# Singleton instance
rag_pipeline = ConversationalRAGPipeline()
