from typing import List
from app.services.chunking.base import BaseChunker


class FixedSizeChunker(BaseChunker):
    """Chunks text into uniform fixed-size windows with a configurable character overlap.

    Attempts to respect whitespace boundaries near the chunk edges so words are not cut in half.
    """

    def chunk(self, text: str) -> List[str]:
        cleaned = text.strip()
        if not cleaned:
            return []

        if len(cleaned) <= self.chunk_size:
            return [cleaned]

        chunks: List[str] = []
        step = self.chunk_size - self.chunk_overlap
        current_idx = 0
        text_len = len(cleaned)

        while current_idx < text_len:
            end_idx = min(current_idx + self.chunk_size, text_len)

            # If not at the very end, try not to cut words in half
            if end_idx < text_len:
                last_space = cleaned.rfind(" ", current_idx, end_idx)
                # Only backtrack if the space is reasonably close to the boundary (within 15% of chunk_size)
                if last_space != -1 and (end_idx - last_space) < (self.chunk_size * 0.2):
                    end_idx = last_space

            chunk_str = cleaned[current_idx:end_idx].strip()
            if chunk_str:
                chunks.append(chunk_str)

            if end_idx >= text_len:
                break

            current_idx += step
            # Ensure forward progress
            if current_idx >= end_idx:
                current_idx = end_idx

        return chunks
