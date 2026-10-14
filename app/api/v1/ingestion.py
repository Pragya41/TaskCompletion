from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AppBaseException, DocumentExtractionError, InvalidFileTypeError
from app.db.session import get_db
from app.schemas.ingestion import (
    ChunkInfo,
    ChunkingStrategyEnum,
    DocumentDetailResponse,
    DocumentIngestResponse,
    DocumentSummaryResponse,
)
from app.services.ingestion.ingestion_service import ingestion_service

router = APIRouter(prefix="/documents", tags=["Document Ingestion"])


@router.post(
    "/ingest",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document (.pdf or .txt) with selectable chunking strategy",
    description=(
        "Uploads a PDF or TXT file, extracts raw text, applies either Fixed-Size or Recursive Character "
        "chunking, creates vector embeddings, indexes them into Qdrant, and saves relational metadata to the SQL database."
    ),
)
async def ingest_document(
    file: UploadFile = File(..., description="Document file (.pdf or .txt)"),
    chunking_strategy: ChunkingStrategyEnum = Form(
        default=ChunkingStrategyEnum.RECURSIVE_CHARACTER,
        description="Selectable chunking strategy: 'fixed_size' or 'recursive_character'",
    ),
    chunk_size: Optional[int] = Form(
        default=500,
        ge=50,
        le=5000,
        description="Maximum characters per chunk",
    ),
    chunk_overlap: Optional[int] = Form(
        default=50,
        ge=0,
        le=1000,
        description="Character overlap between consecutive chunks",
    ),
    db: AsyncSession = Depends(get_db),
) -> DocumentIngestResponse:
    if chunk_overlap and chunk_size and chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_overlap must be strictly less than chunk_size",
        )

    try:
        response = await ingestion_service.ingest_document(
            file=file,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            db=db,
        )
        return response
    except InvalidFileTypeError as e:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(e))
    except DocumentExtractionError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except AppBaseException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during ingestion: {str(e)}",
        )


@router.get(
    "",
    response_model=List[DocumentSummaryResponse],
    summary="List all ingested documents and their metadata",
)
async def list_documents(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
) -> List[DocumentSummaryResponse]:
    docs = await ingestion_service.list_documents(db=db, limit=limit, offset=offset)
    return [DocumentSummaryResponse.model_validate(d) for d in docs]


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Retrieve document metadata and chunks by document ID",
)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> DocumentDetailResponse:
    doc = await ingestion_service.get_document(document_id=document_id, db=db)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )

    chunks_info = [
        ChunkInfo(
            chunk_index=c.chunk_index,
            char_count=c.char_count,
            vector_id=c.vector_id,
            preview=c.chunk_text[:120] + ("..." if len(c.chunk_text) > 120 else ""),
        )
        for c in doc.chunks
    ]

    return DocumentDetailResponse(
        id=doc.id,
        filename=doc.filename,
        file_type=doc.file_type,
        file_size_bytes=doc.file_size_bytes,
        chunking_strategy=doc.chunking_strategy,
        chunk_size=doc.chunk_size,
        chunk_overlap=doc.chunk_overlap,
        chunk_count=doc.chunk_count,
        vector_collection=doc.vector_collection,
        status=doc.status,
        created_at=doc.created_at,
        chunks=chunks_info,
    )
