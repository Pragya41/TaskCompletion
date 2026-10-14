import json
import re
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.exceptions import LLMServiceError
from app.core.logging import logger
from app.schemas.booking import BookingIntentExtraction


class LLMService:
    """Service handling interactions with the LLM for query rewriting, RAG generation, and booking intent extraction."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url or settings.OPENAI_BASE_URL
        self.model = model or settings.LLM_MODEL

        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = None

    async def rewrite_query(self, query: str, chat_history: str) -> str:
        """Rewrites a contextual multi-turn user query into a standalone search query.

        Resolves pronouns and references based on previous conversation turns.
        """
        if not chat_history.strip():
            return query

        if self.client:
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
                logger.warning(f"LLM rewrite query failed ({e}). Using raw query.")

        # Fallback heuristic: If raw query has ambiguous pronouns and history exists, append key context
        return query

    async def extract_booking_intent(
        self, query: str, chat_history: str = ""
    ) -> BookingIntentExtraction:
        """Detects whether the user is trying to book an interview and extracts booking parameters:

        (name, email, date, time).
        """
        full_conversation = f"{chat_history}\nUser: {query}".strip()

        if self.client:
            system_prompt = (
                "You are an interview booking assistant. Analyze the conversation and user's query to determine if they want to book or schedule an interview.\n"
                "Extract the following fields if provided:\n"
                "- name: Full name\n"
                "- email: Email address\n"
                "- date: Date formatted as YYYY-MM-DD (e.g. 2026-10-25)\n"
                "- time: Preferred time (e.g. 14:00 or 2:00 PM)\n\n"
                "Return a strict JSON object with keys:\n"
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
                        {"role": "user", "content": f"Analyze this user query in context:\n{full_conversation}"},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                content = response.choices[0].message.content
                if content:
                    data = json.loads(content)
                    return BookingIntentExtraction(**data)
            except Exception as e:
                logger.warning(f"LLM booking extraction failed ({e}). Falling back to pattern extraction.")

        # Rule-based / regex extraction fallback for zero-API-key testing and robust offline evaluation
        return self._heuristic_extract_booking(query, chat_history)

    def _heuristic_extract_booking(self, query: str, chat_history: str) -> BookingIntentExtraction:
        """Pattern-matching fallback to extract booking intent and parameters."""
        combined_text = f"{chat_history} {query}".lower()
        booking_keywords = ["interview", "book", "schedule", "reserve", "slot", "appointment", "meeting"]
        has_intent = any(keyword in combined_text for keyword in booking_keywords)

        if not has_intent:
            return BookingIntentExtraction(is_booking_intent=False, missing_fields=[])

        text = f"{chat_history}\n{query}"

        # Extract Email
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
        email = email_match.group(0) if email_match else None

        # Extract Date (YYYY-MM-DD or MM/DD/YYYY or words)
        date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        date = date_match.group(1) if date_match else None
        if not date:
            # Check alternative format like 2026/10/20
            alt_date = re.search(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", text)
            if alt_date:
                y, m, d = alt_date.groups()
                date = f"{y}-{int(m):02d}-{int(d):02d}"

        # Extract Time
        time_match = re.search(
            r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)|\d{1,2}:\d{2})\b", text
        )
        time = time_match.group(1) if time_match else None

        # Extract Name (heuristic: 'name is [Name]' or 'I am [Name]')
        name = None
        name_patterns = [
            r"(?:my name is|i am|name:)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
            r"(?:for candidate|candidate:?)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
        ]
        for pattern in name_patterns:
            n_match = re.search(pattern, text, re.IGNORECASE)
            if n_match:
                name = n_match.group(1).strip()
                break

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
        """Custom RAG response generator.

        Synthesizes an accurate answer grounded strictly in retrieved chunks and conversational context.
        """
        if not context_chunks and not chat_history:
            return (
                "I don't have enough context in the knowledge base to answer that question. "
                "Please upload a relevant document or specify your query."
            )

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

        if self.client:
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
                logger.warning(f"OpenAI completion call failed ({e}). Falling back to synthesized answer.")
                if not settings.MOCK_LLM_IF_NO_KEY:
                    raise LLMServiceError(f"LLM generation failed: {str(e)}") from e

        # Offline / Mock intelligent answer generator
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


# Global singleton instance
llm_service = LLMService()
