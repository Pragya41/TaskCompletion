from typing import Union
from app.core.exceptions import ChunkingError
from app.schemas.ingestion import ChunkingStrategyEnum
from app.services.chunking.base import BaseChunker
from app.services.chunking.fixed_size import FixedSizeChunker
from app.services.chunking.recursive_character import RecursiveCharacterChunker


def get_chunker(
    strategy: Union[ChunkingStrategyEnum, str],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> BaseChunker:
    """Factory creating the appropriate chunker instance based on strategy."""
    strat = ChunkingStrategyEnum(strategy) if isinstance(strategy, str) else strategy

    if strat == ChunkingStrategyEnum.FIXED_SIZE:
        return FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strat == ChunkingStrategyEnum.RECURSIVE_CHARACTER:
        return RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        raise ChunkingError(f"Unknown chunking strategy: {strategy}")


__all__ = [
    "BaseChunker",
    "FixedSizeChunker",
    "RecursiveCharacterChunker",
    "get_chunker",
    "ChunkingStrategyEnum",
]
