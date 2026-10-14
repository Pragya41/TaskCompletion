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
    """Manages multi-turn conversation memory stored in Redis."""

    def __init__(self):
        self._client: Optional[aioredis.Redis] = None
        self._is_fake = False

    async def get_client(self) -> aioredis.Redis:
        """Returns the active Redis client, lazily connecting or falling back to in-memory fake."""
        if self._client is not None:
            return self._client

        try:
            client = aioredis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=2.0,
            )
            # Test connection
            await client.ping()
            logger.info(f"Connected to Redis at {settings.REDIS_URL}")
            self._client = client
            self._is_fake = False
            return self._client
        except Exception as e:
            if settings.ALLOW_IN_MEMORY_REDIS_FALLBACK:
                logger.warning(
                    f"Could not connect to Redis server ({e}). Utilizing in-memory FakeRedis for seamless development/testing."
                )
                self._client = fakeredis.aioredis.FakeRedis(decode_responses=True)
                self._is_fake = True
                return self._client
            else:
                logger.error(f"Failed to connect to Redis: {e}", exc_info=True)
                raise RedisMemoryError(f"Redis connection failed: {str(e)}") from e

    def _get_key(self, session_id: str) -> str:
        return f"palm_mind:chat:{session_id}:messages"

    async def add_message(
        self,
        session_id: str,
        role: ChatRole,
        content: str,
    ) -> ChatMessage:
        """Appends a new message to the session's conversational history."""
        client = await self.get_client()
        key = self._get_key(session_id)

        msg = ChatMessage(
            role=role,
            content=content.strip(),
            timestamp=datetime.now(timezone.utc),
        )

        try:
            payload = json.dumps(
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                }
            )
            await client.rpush(key, payload)
            # Refresh TTL
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
        """Retrieves the most recent messages for a session."""
        client = await self.get_client()
        key = self._get_key(session_id)

        try:
            # Fetch last 'limit' messages
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
        """Returns dialogue formatted as 'User: ... \nAssistant: ...' for LLM context."""
        messages = await self.get_messages(session_id, limit=limit)
        if not messages:
            return ""

        formatted_lines = []
        for msg in messages:
            role_label = "User" if msg.role == ChatRole.USER else "Assistant"
            formatted_lines.append(f"{role_label}: {msg.content}")

        return "\n".join(formatted_lines)

    async def clear_history(self, session_id: str) -> bool:
        """Clears all conversation memory for the given session."""
        client = await self.get_client()
        key = self._get_key(session_id)
        try:
            res = await client.delete(key)
            return bool(res)
        except Exception as e:
            logger.error(f"Error deleting chat history for session {session_id}: {e}")
            raise RedisMemoryError(f"Failed to clear history from Redis: {str(e)}") from e

    async def close(self) -> None:
        """Closes the Redis client connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# Global singleton instance
chat_memory = RedisChatMemory()
