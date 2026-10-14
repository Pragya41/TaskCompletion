import json
import re
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.exceptions import AppBaseException, LLMServiceError
from app.core.logging import logger
from app.schemas.booking import BookingIntentExtraction


class LLMService:
    """Handles LLM operations: query rewriting, interview booking extraction, and RAG answering."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key: Optional[str] = api_key or settings.OPENAI_API_KEY
        self.base_url: Optional[str] = base_url or settings.OPENAI_BASE_URL
        self.model: str = model or settings.LLM_MODEL

        # Type explicitly as Optional[AsyncOpenAI] to resolve mypy error
        self.client: Optional[AsyncOpenAI] = None
        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def rewrite_query(self, query: str, chat_history: str) -> str:
        """Rewrite a follow-up query into a standalone question using chat history."""
        if not chat_history.strip():
            return query

        if self.client is not None:
            prompt = (
                "You are an AI search query optimizer. Given the following conversation history and a user's latest query, "
                "rephrase the latest query into a standalone, clear search query that incorporates necessary context from the history. "
                "Do NOT answer the question. Only output the rephrased query. If the query is already standalone or is a booking/greeting, output it unchanged.\n\n"
                f"Conversation History:\n{chat_history}\n\n"
                f"Latest Query: {query}\n\n"
                "Standalone Query:"
            )
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=150,
                )
                rewritten = response.choices[0].message.content
                if rewritten and rewritten.strip():
                    return rewritten.strip().strip('"')
            except Exception as e:
                logger.error(f"LLM rewrite query failed: {e}", exc_info=True)
                if not settings.MOCK_LLM_IF_NO_KEY:
                    raise LLMServiceError(f"Query rewriting failed: {str(e)}") from e

        return query

    async def extract_booking_intent(
        self,
        query: str,
        is_booking_in_progress: bool = False,
        chat_history: str = "",
    ) -> BookingIntentExtraction:
        """Detect if user is booking an interview.

        Fixes Bug 4: Does NOT scan the whole chat history for booking intent unless a booking is currently in progress.
        If a booking is not in progress, only the user's latest query determines booking intent.
        """
        # Determine context to analyze
        if is_booking_in_progress:
            # Booking is in progress (assistant just asked for missing details); check combined context
            context_to_analyze = f"{chat_history}\nUser: {query}".strip()
        else:
            # No booking in progress; only examine the latest user query for intent
            context_to_analyze = query.strip()

        # 1. Real LLM structured extraction via JSON mode
        if self.client is not None:
            system_prompt = (
                "You are an interview booking assistant. Analyze the user's input to determine if they want to book or schedule an interview, "
                "or are answering a previous question about booking details (name, email, date, time).\n"
                f"Booking in progress state: {is_booking_in_progress}\n\n"
                "Extract the following fields if provided:\n"
                "- name: Full candidate name\n"
                "- email: Valid email address\n"
                "- date: Date formatted as YYYY-MM-DD\n"
                "- time: Preferred time (e.g. 14:00 or 2:00 PM)\n\n"
                "Return a strict JSON object:\n"
                "{\n"
                '  "is_booking_intent": boolean,\n'
                '  "name": string or null,\n'
                '  "email": string or null,\n'
                '  "date": string or null,\n'
                '  "time": string or null,\n'
                '  "missing_fields": list of missing fields among ["name", "email", "date", "time"] if is_booking_intent is true\n'
                "}"
            )
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Input:\n{context_to_analyze}"},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                content = response.choices[0].message.content
                if content:
                    data = json.loads(content)
                    return BookingIntentExtraction(**data)
            except Exception as e:
                logger.error(f"LLM booking extraction failed: {e}", exc_info=True)
                if not settings.MOCK_LLM_IF_NO_KEY:
                    raise LLMServiceError(f"Interview booking extraction failed: {str(e)}") from e

        # 2. Heuristic fallback (for test suite or zero-key mode)
        return self._heuristic_extract_booking(query, is_booking_in_progress, chat_history)

    def _heuristic_extract_booking(
        self,
        query: str,
        is_booking_in_progress: bool,
        chat_history: str,
    ) -> BookingIntentExtraction:
        """Deterministic pattern matching to extract booking info without false-triggering on old history."""
        booking_keywords = ["interview", "book", "schedule", "reserve", "slot", "appointment", "meeting"]
        has_direct_intent = any(keyword in query.lower() for keyword in booking_keywords)

        # Only trigger booking intent if current query has intent OR booking was already in progress
        if not has_direct_intent and not is_booking_in_progress:
            return BookingIntentExtraction(is_booking_intent=False, missing_fields=[])

        # Text to search for parameters: include recent history if booking was in progress
        text = f"{chat_history}\n{query}" if is_booking_in_progress else query

        # 1. Extract email
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
        email = email_match.group(0) if email_match else None

        # 2. Extract date (YYYY-MM-DD or YYYY/MM/DD)
        date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        date = date_match.group(1) if date_match else None
        if not date:
            alt_date = re.search(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", text)
            if alt_date:
                y, m, d = alt_date.groups()
                date = f"{y}-{int(m):02d}-{int(d):02d}"

        # 3. Extract time (e.g. 14:00, 2:00 PM)
        time_match = re.search(
            r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)|\d{1,2}:\d{2})\b", text
        )
        time = time_match.group(1) if time_match else None

        # 4. Extract candidate name
        name = None
        name_patterns = [
            r"(?:my name is|i am|name:)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
            r"(?:for candidate|candidate:?)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
            r"\bfor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        ]
        for pattern in name_patterns:
            n_match = re.search(pattern, text, re.IGNORECASE)
            if n_match:
                name = n_match.group(1).strip()
                break

        # Check which required fields are missing
        missing = []
        if not name:
            missing.append("name")
        if not email:
            missing.append("email")
        if not date:
            missing.append("date")
        if not time:
            missing.append("time")

        return BookingIntentExtraction(
            is_booking_intent=True,
            name=name,
            email=email,
            date=date,
            time=time,
            missing_fields=missing,
        )

    async def generate_rag_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        chat_history: str = "",
    ) -> str:
        """Generate answer grounded in retrieved document chunks with citations."""
        if not context_chunks and not chat_history:
            return (
                "I don't have enough context in the knowledge base to answer that question. "
                "Please upload a relevant document or specify your query."
            )

        # Build context string with numbered citations
        formatted_context = ""
        for i, chunk in enumerate(context_chunks, 1):
            filename = chunk.get("filename", "unknown")
            chunk_idx = chunk.get("chunk_index", 0)
            text = chunk.get("text", "")
            formatted_context += f"--- Source [{i}] (File: {filename}, Chunk: {chunk_idx}) ---\n{text}\n\n"

        system_prompt = (
            "You are Palm Mind AI, an intelligent, professional technical assistant. "
            "Answer the user's inquiry based strictly on the retrieved context below. "
            "If the answer cannot be found in the context, state that clearly rather than hallucinating. "
            "When referencing information, cite the corresponding Source number [1], [2], etc.\n\n"
            f"Retrieved Context:\n{formatted_context if formatted_context else 'No document context available.'}"
        )

        user_content = ""
        if chat_history.strip():
            user_content += f"Conversation History:\n{chat_history}\n\n"
        user_content += f"User Question: {query}"

        # Call real OpenAI LLM
        if self.client is not None:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.2,
                )
                answer = response.choices[0].message.content
                if answer and answer.strip():
                    return answer.strip()
            except Exception as e:
                logger.error(f"OpenAI completion call failed: {e}", exc_info=True)
                if not settings.MOCK_LLM_IF_NO_KEY:
                    raise LLMServiceError(f"OpenAI generation failed: {str(e)}") from e

        # Mock fallback answer only for offline testing
        if settings.MOCK_LLM_IF_NO_KEY:
            if context_chunks:
                top_chunk = context_chunks[0]
                sources_cited = ", ".join(f"[{i+1}]" for i in range(len(context_chunks)))
                return (
                    f"Based on the provided documents {sources_cited}:\n\n"
                    f"{top_chunk.get('text', '').strip()[:350]}...\n\n"
                    f"(Retrieved from '{top_chunk.get('filename')}', chunk index {top_chunk.get('chunk_index')})"
                )
            else:
                return "Hello! I am Palm Mind AI assistant. How may I assist you with your documents or interview booking today?"

        raise AppBaseException(
            "OPENAI_API_KEY is not configured. Please supply a valid OpenAI API key in your .env file."
        )


# Singleton instance
llm_service = LLMService()
