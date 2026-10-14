from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ChunkingStrategyEnum(str, Enum):
    FIXED_SIZE = "fixed_size"
    RECURSIVE_CHARACTER = "recursive_character"


class ChunkInfo(BaseModel):
    chunk_index: int
    char_count: int
    vector_id: str
    preview: str


class DocumentIngestResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    chunking_strategy: ChunkingStrategyEnum
    chunk_size: int
    chunk_overlap: int
    total_chunks: int
    vector_collection: str
    status: str
    created_at: datetime
    sample_chunks: List[ChunkInfo] = []


class DocumentSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_type: str
    file_size_bytes: int
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    chunk_count: int
    vector_collection: str
    status: str
    created_at: datetime


class DocumentDetailResponse(DocumentSummaryResponse):
    chunks: List[ChunkInfo] = []
