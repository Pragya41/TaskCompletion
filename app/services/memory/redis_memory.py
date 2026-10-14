import json
from datetime import datetime, timezone
from typing import List, Optional
import redis.asyncio as aioredis
import fakeredis.aioredis

from app.core.config import settings
from app.core.exceptions import RedisMemoryError
from app.core.logging import logger
from app.schemas.chat import ChatMessage, ChatRole


class RedisChatMemory:
    """Manages chat history and session dialogue state in Redis with automatic TTL and fakeredis fallback."""

    def __init__(self):
        self._client: Optional[aioredis.Redis] = None
        self._is_fake = False

    async def get_client(self) -> aioredis.Redis:
        """Connect to Redis, or use in-memory fakeredis if Redis is not running."""
        if self._client is not None:
            return self._client

        try:
            # Try connecting to standalone Redis
            client = aioredis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=2.0,
            )
            await client.ping()
            logger.info(f"Connected to Redis at {settings.REDIS_URL}")
            self._client = client
            self._is_fake = False
            return self._client

        except Exception as e:
            # Fallback to fakeredis for zero-dependency local testing
            if settings.ALLOW_IN_MEMORY_REDIS_FALLBACK:
                logger.warning(
                    f"Could not connect to Redis server ({e}). Using in-memory FakeRedis for local development."
                )
                self._client = fakeredis.aioredis.FakeRedis(decode_responses=True)
                self._is_fake = True
                return self._client
            else:
                logger.error(f"Failed to connect to Redis: {e}", exc_info=True)
                raise RedisMemoryError(f"Redis connection failed: {str(e)}") from e

    def _get_key(self, session_id: str) -> str:
        """Redis key for session messages."""
        return f"palm_mind:chat:{session_id}:messages"

    def _get_state_key(self, session_id: str) -> str:
        """Redis key for session state (e.g. BOOKING_IN_PROGRESS)."""
        return f"palm_mind:chat:{session_id}:state"

    async def add_message(
        self,
        session_id: str,
        role: ChatRole,
        content: str,
    ) -> ChatMessage:
        """Save a new user or assistant message to Redis."""
        client = await self.get_client()
        key = self._get_key(session_id)

        msg = ChatMessage(
            role=role,
            content=content.strip(),
            timestamp=datetime.now(timezone.utc),
        )

        try:
            # Convert to JSON and push to Redis list
            payload = json.dumps(
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                }
            )
            await client.rpush(key, payload)

            # Refresh TTL on the session key
            await client.expire(key, settings.REDIS_CHAT_TTL_SECONDS)
            return msg

        except Exception as e:
            logger.error(f"Error persisting message to Redis for session {session_id}: {e}")
            raise RedisMemoryError(f"Failed to save message to Redis: {str(e)}") from e

    async def get_messages(
        self,
        session_id: str,
        limit: int = 10,
    ) -> List[ChatMessage]:
        """Get the most recent messages for a session."""
        client = await self.get_client()
        key = self._get_key(session_id)

        try:
            # Get last 'limit' messages using negative indices
            start = -limit if limit > 0 else 0
            raw_items = await client.lrange(key, start, -1)
            messages: List[ChatMessage] = []

            for item in raw_items:
                data = json.loads(item)
                messages.append(
                    ChatMessage(
                        role=ChatRole(data["role"]),
                        content=data["content"],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                    )
                )
            return messages

        except Exception as e:
            logger.error(f"Error reading chat history from Redis for session {session_id}: {e}")
            raise RedisMemoryError(f"Failed to read messages from Redis: {str(e)}") from e

    async def get_formatted_history(self, session_id: str, limit: int = 6) -> str:
        """Format history as 'User: ... \\nAssistant: ...' for the LLM."""
        messages = await self.get_messages(session_id, limit=limit)
        if not messages:
            return ""

        formatted_lines = []
        for msg in messages:
            role_label = "User" if msg.role == ChatRole.USER else "Assistant"
            formatted_lines.append(f"{role_label}: {msg.content}")

        return "\n".join(formatted_lines)

    async def get_session_state(self, session_id: str) -> Optional[str]:
        """Retrieves conversational state flag from Redis (e.g. 'BOOKING_IN_PROGRESS')."""
        client = await self.get_client()
        key = self._get_state_key(session_id)
        try:
            val = await client.get(key)
            return str(val) if val is not None else None
        except Exception as e:
            logger.warning(f"Error reading session state for {session_id}: {e}")
            return None

    async def set_session_state(self, session_id: str, state: str) -> None:
        """Persists conversational state flag in Redis with TTL."""
        client = await self.get_client()
        key = self._get_state_key(session_id)
        try:
            await client.set(key, state, ex=settings.REDIS_CHAT_TTL_SECONDS)
        except Exception as e:
            logger.warning(f"Error setting session state for {session_id}: {e}")

    async def clear_session_state(self, session_id: str) -> None:
        """Deletes conversational state flag from Redis."""
        client = await self.get_client()
        key = self._get_state_key(session_id)
        try:
            await client.delete(key)
        except Exception as e:
            logger.warning(f"Error clearing session state for {session_id}: {e}")

    async def clear_history(self, session_id: str) -> bool:
        """Delete all chat history and state for a session."""
        client = await self.get_client()
        key = self._get_key(session_id)
        state_key = self._get_state_key(session_id)
        try:
            await client.delete(state_key)
            res = await client.delete(key)
            return bool(res)
        except Exception as e:
            logger.error(f"Error deleting chat history for session {session_id}: {e}")
            raise RedisMemoryError(f"Failed to clear history from Redis: {str(e)}") from e

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# Singleton instance
chat_memory = RedisChatMemory()
