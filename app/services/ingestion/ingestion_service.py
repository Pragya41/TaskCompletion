import uuid
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import (
    AppBaseException,
    DocumentExtractionError,
    InvalidFileTypeError,
)
from app.core.logging import logger
from app.models.document import Document, DocumentChunk
from app.schemas.ingestion import (
    ChunkInfo,
    ChunkingStrategyEnum,
    DocumentIngestResponse,
)
from app.services.chunking import get_chunker
from app.services.embeddings import embedding_service
from app.services.extraction import TextExtractor
from app.services.vector_store import vector_store


class DocumentIngestionService:
    """Handles file upload, text extraction, chunking, embeddings, and database storage."""

    async def ingest_document(
        self,
        file: UploadFile,
        chunking_strategy: ChunkingStrategyEnum,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        db: Optional[AsyncSession] = None,
    ) -> DocumentIngestResponse:
        filename = file.filename or "unknown_file.txt"
        file_ext = "." + filename.split(".")[-1].lower() if "." in filename else ""

        # Validate file format against allowed extensions (.pdf, .txt)
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise InvalidFileTypeError(
                f"File '{filename}' has unsupported extension '{file_ext}'. "
                f"Allowed extensions are: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )

        # Enforce file boundaries: read content and reject empty files or files exceeding max size limit
        content = await file.read()
        file_size_bytes = len(content)

        if file_size_bytes == 0:
            raise DocumentExtractionError(f"Uploaded file '{filename}' is empty.")

        if file_size_bytes > settings.MAX_FILE_SIZE_BYTES:
            raise DocumentExtractionError(
                f"File size exceeds maximum limit of {settings.MAX_FILE_SIZE_BYTES} bytes."
            )

        # Text extraction:
        # Route file content to the appropriate extractor (pypdf for PDF, multi-encoding decoder for TXT)
        extracted_text, detected_type = TextExtractor.extract_text(filename, content)
        logger.info(f"Extracted {len(extracted_text)} characters from '{filename}'")

        # Chunking strategy execution:
        # Instantiate either FixedSizeChunker or RecursiveCharacterChunker based on the caller's request,
        # using specified or configured default chunk_size and chunk_overlap
        c_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
        c_overlap = chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP

        chunker = get_chunker(
            strategy=chunking_strategy,
            chunk_size=c_size,
            chunk_overlap=c_overlap,
        )
        text_chunks = chunker.chunk(extracted_text)

        if not text_chunks:
            raise DocumentExtractionError("No valid text chunks were produced from the document.")

        # Vector embedding generation:
        # Embed all extracted chunks concurrently using OpenAI text-embedding-3-small
        embeddings = await embedding_service.embed_texts(text_chunks)

        # Dual-write preparation:
        # Build vector payloads for Qdrant and relational entity models for PostgreSQL/SQLite
        # Each chunk is linked to the parent document via document_id and stores its chunk_index
        doc_id = str(uuid.uuid4())
        col_name = settings.QDRANT_COLLECTION_NAME

        qdrant_payloads = []
        doc_chunk_models: List[DocumentChunk] = []
        sample_chunks_info: List[ChunkInfo] = []

        for idx, (chunk_text, vector) in enumerate(zip(text_chunks, embeddings)):
            vector_id = str(uuid.uuid4())

            # Payload indexed directly into Qdrant vector database
            qdrant_payloads.append(
                {
                    "vector_id": vector_id,
                    "vector": vector,
                    "document_id": doc_id,
                    "filename": filename,
                    "chunk_index": idx,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                }
            )

            # Relational database record mapping the chunk text and vector ID
            chunk_record = DocumentChunk(
                id=str(uuid.uuid4()),
                document_id=doc_id,
                chunk_index=idx,
                chunk_text=chunk_text,
                char_count=len(chunk_text),
                vector_id=vector_id,
            )
            doc_chunk_models.append(chunk_record)

            # Store preview of initial chunks to return in the ingestion response
            if idx < 3:
                sample_chunks_info.append(
                    ChunkInfo(
                        chunk_index=idx,
                        char_count=len(chunk_text),
                        vector_id=vector_id,
                        preview=chunk_text[:120] + ("..." if len(chunk_text) > 120 else ""),
                    )
                )

        # Index vectors and metadata into Qdrant collection
        await vector_store.upsert_chunks(qdrant_payloads, collection_name=col_name)

        # Commit document metadata and chunk records to the relational database
        document_record = Document(
            id=doc_id,
            filename=filename,
            file_type=detected_type,
            file_size_bytes=file_size_bytes,
            chunking_strategy=chunking_strategy.value,
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            chunk_count=len(text_chunks),
            vector_collection=col_name,
            status="PROCESSED",
        )

        if db is not None:
            db.add(document_record)
            for c_model in doc_chunk_models:
                db.add(c_model)
            await db.commit()
            await db.refresh(document_record)

        return DocumentIngestResponse(
            document_id=doc_id,
            filename=filename,
            file_type=detected_type,
            file_size_bytes=file_size_bytes,
            chunking_strategy=chunking_strategy,
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            total_chunks=len(text_chunks),
            vector_collection=col_name,
            status="PROCESSED",
            created_at=document_record.created_at,
            sample_chunks=sample_chunks_info,
        )

    async def list_documents(
        self,
        db: AsyncSession,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Document]:
        """Fetch all uploaded documents."""
        stmt = (
            select(Document)
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_document(
        self,
        document_id: str,
        db: AsyncSession,
    ) -> Optional[Document]:
        """Fetch a single document by its ID."""
        stmt = select(Document).where(Document.id == document_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()


# Singleton instance
ingestion_service = DocumentIngestionService()
