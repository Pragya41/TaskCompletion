from typing import Optional
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
    """End-to-end custom Conversational RAG pipeline.

    - Uses Redis for conversational memory.
    - Resolves multi-turn query context without relying on RetrievalQAChain.
    - Retrieves top-k semantically similar context from Qdrant.
    - Detects and executes interview booking via LLM tool/intent extraction.
    """

    async def execute(
        self,
        request: ChatRequest,
        db: AsyncSession,
    ) -> ChatResponse:
        session_id = request.session_id
        raw_query = request.query.strip()

        # Step 1: Retrieve prior conversation context from Redis
        chat_history = await chat_memory.get_formatted_history(session_id, limit=6)

        # Step 2: Persist current user message into Redis memory
        await chat_memory.add_message(
            session_id=session_id,
            role=ChatRole.USER,
            content=raw_query,
        )

        # Step 3: Check for Interview Booking intent
        booking_intent = await llm_service.extract_booking_intent(
            query=raw_query,
            chat_history=chat_history,
        )

        if booking_intent.is_booking_intent:
            if not booking_intent.missing_fields and booking_intent.name and booking_intent.email and booking_intent.date and booking_intent.time:
                # All required booking parameters present -> Create booking in DB
                logger.info(
                    f"Interview booking intent confirmed with all parameters for session {session_id}"
                )
                booking_in = BookingCreate(
                    name=booking_intent.name,
                    email=booking_intent.email,
                    date=booking_intent.date,
                    time=booking_intent.time,
                    session_id=session_id,
                    notes=f"Scheduled via Conversational RAG agent. Query: {raw_query}",
                )
                booking_record = await booking_service.create_booking(db=db, booking_in=booking_in)

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

                # Save assistant reply to Redis
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
                # Booking intent present, but missing details -> prompt user
                missing_str = ", ".join(booking_intent.missing_fields)
                clarification_msg = (
                    f"I would be glad to help schedule your interview! To confirm the reservation, "
                    f"could you please provide the following missing details: {missing_str}?"
                )

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

        # Step 4: Multi-turn Query Rewriting
        rewritten_query = await llm_service.rewrite_query(
            query=raw_query,
            chat_history=chat_history,
        )

        # Step 5: Vector Retrieval from Qdrant
        query_vector = await embedding_service.embed_query(rewritten_query)
        retrieved_hits = await vector_store.search(
            query_vector=query_vector,
            top_k=request.top_k,
            document_id=request.document_id,
        )

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

        # Step 6: Custom RAG Answer Generation
        answer = await llm_service.generate_rag_answer(
            query=rewritten_query,
            context_chunks=retrieved_hits,
            chat_history=chat_history,
        )

        # Step 7: Persist assistant answer to Redis memory
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


# Global singleton instance
rag_pipeline = ConversationalRAGPipeline()
