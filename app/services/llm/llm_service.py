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

        # Initialize OpenAI client if key is available
        self.client: Optional[AsyncOpenAI] = None
        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def rewrite_query(self, query: str, chat_history: str) -> str:
        """Conversational query reformulation:
        In multi-turn conversations, follow-up questions often contain pronouns or omissions
        (e.g., 'What are its key benefits?' or 'Tell me more about the first one').
        This method rephrases the query into a standalone question containing full context
        so that downstream vector similarity search finds the most relevant document chunks.
        """
        # Return query directly if there is no conversation history to resolve
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
        """Interview booking extraction:
        Analyzes the user's message to identify scheduling intent and extract candidate fields
        (name, email, date, time).
        
        To prevent historical booking mentions from falsely capturing future normal questions,
        we only incorporate the conversation history if a booking is actively in progress
        (indicated by the session state flag in Redis).
        """
        # Context scoping:
        # If a booking is currently underway, include chat history to resolve details supplied in pieces.
        # Otherwise, only analyze the incoming query to prevent false-positive bookings.
        if is_booking_in_progress:
            context_to_analyze = f"{chat_history}\nUser: {query}".strip()
        else:
            context_to_analyze = query.strip()

        # Call OpenAI structured JSON extraction when client is active
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

        # Fall back to pattern matching if operating in offline test mode
        return self._heuristic_extract_booking(query, is_booking_in_progress, chat_history)

    def _heuristic_extract_booking(
        self,
        query: str,
        is_booking_in_progress: bool,
        chat_history: str,
    ) -> BookingIntentExtraction:
        """Simple pattern matching to extract booking info when LLM is unavailable."""
        booking_keywords = ["interview", "book", "schedule", "reserve", "slot", "appointment", "meeting"]
        has_direct_intent = any(keyword in query.lower() for keyword in booking_keywords)

        # Skip booking extraction if no booking keyword is present and no booking is underway
        if not has_direct_intent and not is_booking_in_progress:
            return BookingIntentExtraction(is_booking_intent=False, missing_fields=[])

        text = f"{chat_history}\n{query}" if is_booking_in_progress else query

        # Parse email address:
        # Use re.findall and pick the last occurrence ([-1]) so that if a user corrects
        # a previously mistyped email in a follow-up turn, the newest email takes precedence
        email_matches = re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", text)
        email = email_matches[-1] if email_matches else None

        # Parse date in YYYY-MM-DD or YYYY/MM/DD formats:
        # Pick the latest date found in the dialogue so candidate date corrections take precedence
        date_matches = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        date = date_matches[-1] if date_matches else None
        if not date:
            alt_dates = re.findall(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", text)
            if alt_dates:
                y, m, d = alt_dates[-1]
                date = f"{y}-{int(m):02d}-{int(d):02d}"

        # Parse time expression:
        # Pick the latest time occurrence to ensure updated time slots override earlier ones
        time_matches = re.findall(
            r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)|\d{1,2}:\d{2})\b", text
        )
        time = time_matches[-1] if time_matches else None

        # Parse candidate name:
        # Search using priority patterns and select the most recent match
        name = None
        name_patterns = [
            r"(?:my name is|i am|name:)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
            r"(?:for candidate|candidate:?)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
            r"\bfor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        ]
        for pattern in name_patterns:
            n_matches = re.findall(pattern, text, re.IGNORECASE)
            if n_matches:
                name = n_matches[-1].strip()
                break

        # Identify any missing mandatory parameters
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
        """Grounded answer generation:
        Implements a custom RAG prompt template (without relying on LangChain's RetrievalQAChain)
        that formats retrieved text chunks into labeled sources [1], [2], instructs the model
        to strictly cite these sources, and enforces zero-hallucination policies if the answer
        is not present in the ingested context.
        """
        if not context_chunks and not chat_history:
            return (
                "I don't have enough context in the knowledge base to answer that question. "
                "Please upload a relevant document or specify your query."
            )

        # Assemble retrieved passages with clear source labels and chunk indices
        formatted_context = ""
        for i, chunk in enumerate(context_chunks, 1):
            filename = chunk.get("filename", "unknown")
            chunk_idx = chunk.get("chunk_index", 0)
            text = chunk.get("text", "")
            formatted_context += f"--- Source [{i}] (File: {filename}, Chunk: {chunk_idx}) ---\n{text}\n\n"

        # Grounded system prompt enforcing citations and preventing hallucinations
        system_prompt = (
            "You are Palm Mind AI, an intelligent, professional technical assistant. "
            "Answer the user's inquiry based strictly on the retrieved context below. "
            "If the answer cannot be found in the context, state that clearly rather than hallucinating. "
            "When referencing information, cite the corresponding Source number [1], [2], etc.\n\n"
            f"Retrieved Context:\n{formatted_context if formatted_context else 'No document context available.'}"
        )

        # Append prior conversation history to maintain contextual continuity
        user_content = ""
        if chat_history.strip():
            user_content += f"Conversation History:\n{chat_history}\n\n"
        user_content += f"User Question: {query}"

        # Send grounded prompt to OpenAI chat completions API
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
