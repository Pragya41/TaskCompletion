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

        # Check whether an interview booking conversation is currently in progress.
        # Storing this state in Redis ensures that previous turns only influence booking
        # if the user is actively answering scheduling questions, preventing normal follow-up
        # queries from being mistakenly treated as bookings.
        session_state = await chat_memory.get_session_state(session_id)
        is_booking_in_progress = (session_state == "BOOKING_IN_PROGRESS")

        # Load recent dialogue history to keep conversational context
        chat_history = await chat_memory.get_formatted_history(session_id, limit=6)

        # Record the user's incoming message in Redis memory
        await chat_memory.add_message(
            session_id=session_id,
            role=ChatRole.USER,
            content=raw_query,
        )

        # Detect whether the user is asking to schedule an interview or replying to a booking question.
        # When not in progress, only the latest query triggers booking intent.
        booking_intent = await llm_service.extract_booking_intent(
            query=raw_query,
            is_booking_in_progress=is_booking_in_progress,
            chat_history=chat_history,
        )

        if booking_intent.is_booking_intent:
            # Check if all four required fields were collected (name, email, date, time)
            if not booking_intent.missing_fields and booking_intent.name and booking_intent.email and booking_intent.date and booking_intent.time:
                # Validate date and time formats before saving.
                # If the candidate provides a past date or invalid time, catch the validation error
                # and return a polite conversational reply instead of raising an internal server error.
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
                    # Extract the specific validation issue (e.g., date cannot be in the past)
                    first_err = val_err.errors()[0]
                    raw_msg = first_err.get("msg", "Please provide a valid future date and time.")
                    clean_msg = raw_msg.replace("Value error, ", "").strip()

                    validation_reply = (
                        f"I noticed an issue with the booking details: {clean_msg} "
                        f"Could you please provide an updated date or time?"
                    )

                    # Keep booking state active in Redis so the user can easily reply with just the corrected date or time
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

                # Persist the confirmed booking in the SQL database
                logger.info(f"Booking interview for candidate {booking_in.name} (session: {session_id})")
                booking_record = await booking_service.create_booking(db=db, booking_in=booking_in)

                # Reset booking state flag since booking is complete.
                # This guarantees that future turns are processed normally as RAG queries.
                await chat_memory.clear_session_state(session_id)

                # Send confirmation details and reference ID back to candidate
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

                # Store assistant confirmation in Redis
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
                # Flag booking as in-progress in Redis and ask user for missing details.
                # This allows candidate to provide remaining fields in subsequent messages.
                await chat_memory.set_session_state(session_id, "BOOKING_IN_PROGRESS")

                missing_str = ", ".join(booking_intent.missing_fields)
                clarification_msg = (
                    f"I would be glad to help schedule your interview! To confirm the reservation, "
                    f"could you please provide the following missing details: {missing_str}?"
                )

                # Store question in Redis
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

        # Clear any stale booking state when user is asking a standard document question
        if is_booking_in_progress:
            await chat_memory.clear_session_state(session_id)

        # Contextual query reformulation:
        # Rephrase the incoming query taking past dialogue into account to resolve
        # pronouns and references (e.g., 'What is its pricing?' -> 'What is the pricing of Palm Mind AI?')
        rewritten_query = await llm_service.rewrite_query(
            query=raw_query,
            chat_history=chat_history,
        )

        # Vector similarity search in Qdrant:
        # Convert rewritten question into an embedding and retrieve the top-k most relevant text chunks
        query_vector = await embedding_service.embed_query(rewritten_query)
        retrieved_hits = await vector_store.search(
            query_vector=query_vector,
            top_k=request.top_k,
            document_id=request.document_id,
        )

        # Attach source metadata for document attribution
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

        # Grounded answer synthesis:
        # Custom RAG implementation (strictly no RetrievalQAChain) that injects retrieved chunks
        # into the system prompt and asks the LLM to generate an answer with exact citations
        answer = await llm_service.generate_rag_answer(
            query=rewritten_query,
            context_chunks=retrieved_hits,
            chat_history=chat_history,
        )

        # Save assistant answer to memory for future conversational turns
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
