from typing import List
from app.services.chunking.base import BaseChunker


class FixedSizeChunker(BaseChunker):
    """Chunking Strategy 1: Fixed-size windows with character overlap.

    Also avoids splitting words in half by snapping to nearby spaces.
    """

    def chunk(self, text: str) -> List[str]:
        cleaned = text.strip()
        if not cleaned:
            return []

        # If text is already small enough, return as one chunk
        if len(cleaned) <= self.chunk_size:
            return [cleaned]

        chunks: List[str] = []
        step = self.chunk_size - self.chunk_overlap
        current_idx = 0
        text_len = len(cleaned)

        # Slide window across text
        while current_idx < text_len:
            end_idx = min(current_idx + self.chunk_size, text_len)

            # Try not to cut words in half: find last space near the end
            if end_idx < text_len:
                last_space = cleaned.rfind(" ", current_idx, end_idx)
                if last_space != -1 and (end_idx - last_space) < (self.chunk_size * 0.2):
                    end_idx = last_space

            # Save the chunk
            chunk_str = cleaned[current_idx:end_idx].strip()
            if chunk_str:
                chunks.append(chunk_str)

            # Stop if we reached the end
            if end_idx >= text_len:
                break

            # Move window forward
            current_idx += step
            if current_idx >= end_idx:
                current_idx = end_idx

        return chunks
