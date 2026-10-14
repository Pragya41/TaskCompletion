import uuid
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.config import settings
from app.core.exceptions import VectorStoreError
from app.core.logging import logger


class QdrantVectorStore:
    """Handles storing and searching vectors in Qdrant (remote or embedded)."""

    def __init__(self):
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        """Get or initialize the Qdrant client."""
        if self._client is None:
            self._client = self._init_client()
        return self._client

    def _init_client(self) -> QdrantClient:
        """Connect to remote Qdrant URL if provided, otherwise use local disk storage."""
        if settings.QDRANT_URL:
            logger.info(f"Connecting to remote Qdrant at: {settings.QDRANT_URL}")
            return QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=30.0,
            )
        else:
            logger.info(f"Using embedded Qdrant storage at: {settings.QDRANT_EMBEDDED_PATH}")
            return QdrantClient(path=settings.QDRANT_EMBEDDED_PATH)

    def ensure_collection(
        self,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> None:
        """Create the Qdrant collection with Cosine distance if it doesn't exist."""
        col_name = collection_name or settings.QDRANT_COLLECTION_NAME
        dim = dimension or settings.VECTOR_DIMENSION

        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == col_name for c in collections)
            if not exists:
                logger.info(f"Creating Qdrant collection '{col_name}' (dim={dim}, distance=COSINE)")
                self.client.create_collection(
                    collection_name=col_name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection '{col_name}': {e}", exc_info=True)
            raise VectorStoreError(f"Failed to initialize Qdrant collection: {str(e)}") from e

    async def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> List[str]:
        """Save chunk vectors and metadata into Qdrant."""
        if not chunks:
            return []

        col_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.ensure_collection(col_name)

        points: List[PointStruct] = []
        vector_ids: List[str] = []

        # Build PointStruct for each chunk
        for chunk in chunks:
            point_id = chunk.get("vector_id") or str(uuid.uuid4())
            vector_ids.append(point_id)

            payload = {
                "document_id": chunk.get("document_id"),
                "filename": chunk.get("filename"),
                "chunk_index": chunk.get("chunk_index"),
                "text": chunk.get("text"),
                "char_count": chunk.get("char_count", len(chunk.get("text", ""))),
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=chunk["vector"],
                    payload=payload,
                )
            )

        # Batch upsert to Qdrant
        try:
            self.client.upsert(
                collection_name=col_name,
                points=points,
                wait=True,
            )
            return vector_ids
        except Exception as e:
            logger.error(f"Failed to upsert points into Qdrant: {e}", exc_info=True)
            raise VectorStoreError(f"Qdrant upsert failed: {str(e)}") from e

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        document_id: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search Qdrant for the top_k most similar chunks to the query vector."""
        col_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.ensure_collection(col_name)

        # Optional filter by document_id
        query_filter = None
        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )

        # Run vector query
        try:
            response = self.client.query_points(
                collection_name=col_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )

            results: List[Dict[str, Any]] = []
            for hit in response.points:
                payload = hit.payload or {}
                results.append(
                    {
                        "vector_id": str(hit.id),
                        "score": float(hit.score),
                        "document_id": payload.get("document_id", ""),
                        "filename": payload.get("filename", ""),
                        "chunk_index": payload.get("chunk_index", 0),
                        "text": payload.get("text", ""),
                    }
                )
            return results
        except Exception as e:
            logger.error(f"Error querying Qdrant: {e}", exc_info=True)
            raise VectorStoreError(f"Qdrant query failed: {str(e)}") from e

    async def delete_by_document_id(
        self,
        document_id: str,
        collection_name: Optional[str] = None,
    ) -> None:
        """Delete all chunks belonging to a document."""
        col_name = collection_name or settings.QDRANT_COLLECTION_NAME
        try:
            self.client.delete(
                collection_name=col_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
            )
        except Exception as e:
            logger.warning(f"Could not delete Qdrant points for doc {document_id}: {e}")


# Singleton instance
vector_store = QdrantVectorStore()
