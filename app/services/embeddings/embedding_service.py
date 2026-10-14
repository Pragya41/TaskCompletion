import hashlib
from typing import List, Optional
import numpy as np
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.exceptions import AppBaseException
from app.core.logging import logger


class EmbeddingService:
    """Handles generating vector embeddings for text chunks and queries."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        self.api_key: Optional[str] = api_key or settings.OPENAI_API_KEY
        self.base_url: Optional[str] = base_url or settings.OPENAI_BASE_URL
        self.model: str = model or settings.EMBEDDING_MODEL
        self.dimension: int = dimension or settings.VECTOR_DIMENSION

        # Initialize OpenAI async client
        self.client: Optional[AsyncOpenAI] = None
        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of text chunks using OpenAI."""
        if not texts:
            return []

        # Request embeddings from OpenAI if client is configured
        if self.client is not None:
            try:
                response = await self.client.embeddings.create(
                    input=texts,
                    model=self.model,
                )
                return [data.embedding for data in response.data]
            except Exception as e:
                logger.error(f"OpenAI embedding generation failed: {e}", exc_info=True)
                # Fail loudly to prevent storing fake vectors alongside real embeddings
                raise AppBaseException(f"Failed to generate OpenAI embeddings: {str(e)}") from e

        # Generate deterministic unit vectors when running offline test suite
        if settings.MOCK_LLM_IF_NO_KEY:
            logger.debug("Generating deterministic unit embeddings for offline testing.")
            return [self._generate_deterministic_vector(text) for text in texts]

        # Require a valid API key if not in mock test mode
        raise AppBaseException(
            "OPENAI_API_KEY is not configured. Please supply a valid OpenAI API key in your .env file "
            "to generate embeddings with text-embedding-3-small."
        )

    async def embed_query(self, query: str) -> List[float]:
        """Generates an embedding vector for a search query."""
        results = await self.embed_texts([query])
        return results[0]

    def _generate_deterministic_vector(self, text: str) -> List[float]:
        """Generates a pseudo-random, deterministic unit vector from text tokens for offline testing."""
        vector = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        if not words:
            words = ["empty"]

        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 4) % 2 == 0) else -1.0
            weight = 1.0 + (len(word) / 10.0)
            vector[idx] += sign * weight

        # Normalize to unit length for cosine similarity
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        else:
            vector[0] = 1.0

        return vector.tolist()


# Global singleton instance
embedding_service = EmbeddingService()
