import pytest
from app.schemas.ingestion import ChunkingStrategyEnum
from app.services.chunking import get_chunker
from app.services.chunking.fixed_size import FixedSizeChunker
from app.services.chunking.recursive_character import RecursiveCharacterChunker


def test_fixed_size_chunker():
    text = "Word " * 100  # 500 characters
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(text)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 120  # Allows slight margin for word boundary snapping
        assert len(chunk) > 0


def test_recursive_character_chunker():
    paragraphs = [
        "Paragraph 1: Palm Mind AI develops scalable AI architectures.",
        "Paragraph 2: The Conversational RAG engine uses Redis memory.",
        "Paragraph 3: Interview scheduling is completely automated.",
    ]
    full_text = "\n\n".join(paragraphs)

    chunker = RecursiveCharacterChunker(chunk_size=120, chunk_overlap=20)
    chunks = chunker.chunk(full_text)

    assert len(chunks) >= 3
    assert any("Paragraph 1" in c for c in chunks)
    assert any("Paragraph 2" in c for c in chunks)
    assert any("Paragraph 3" in c for c in chunks)


def test_chunker_factory():
    c1 = get_chunker(ChunkingStrategyEnum.FIXED_SIZE, chunk_size=200, chunk_overlap=20)
    assert isinstance(c1, FixedSizeChunker)

    c2 = get_chunker(ChunkingStrategyEnum.RECURSIVE_CHARACTER, chunk_size=200, chunk_overlap=20)
    assert isinstance(c2, RecursiveCharacterChunker)


def test_invalid_chunk_overlap():
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=100, chunk_overlap=100)
