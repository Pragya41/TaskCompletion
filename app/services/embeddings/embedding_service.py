import hashlib
import math
from typing import List, Optional
import numpy as np

from openai import AsyncOpenAI
from app.core.config import settings
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
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url or settings.OPENAI_BASE_URL
        self.model = model or settings.EMBEDDING_MODEL
        self.dimension = dimension or settings.VECTOR_DIMENSION

        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = None

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of text chunks."""
        if not texts:
            return []

        if self.client:
            try:
                response = await self.client.embeddings.create(
                    input=texts,
                    model=self.model,
                )
                return [data.embedding for data in response.data]
            except Exception as e:
                logger.warning(
                    f"OpenAI embedding call failed ({e}). Falling back to deterministic embedding generation."
                )
                if not settings.MOCK_LLM_IF_NO_KEY:
                    raise

        # Deterministic mock embeddings for offline / zero-key testing
        return [self._generate_deterministic_vector(text) for text in texts]

    async def embed_query(self, query: str) -> List[float]:
        """Generates an embedding vector for a search query."""
        results = await self.embed_texts([query])
        return results[0]

    def _generate_deterministic_vector(self, text: str) -> List[float]:
        """Generates a pseudo-random, deterministic unit vector from text tokens.

        Allows testing vector search and cosine similarity offline without API keys.
        """
        vector = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        if not words:
            words = ["empty"]

        for word in words:
            # Hash word into pseudo-random index and sign
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
