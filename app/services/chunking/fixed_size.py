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

        # Return immediately if the document text is already smaller than or equal to chunk_size
        if len(cleaned) <= self.chunk_size:
            return [cleaned]

        chunks: List[str] = []
        # Calculate step distance: how much we advance on each iteration
        # The overlap is preserved because step is smaller than chunk_size
        step = self.chunk_size - self.chunk_overlap
        current_idx = 0
        text_len = len(cleaned)

        # Slide the window across the text content
        while current_idx < text_len:
            end_idx = min(current_idx + self.chunk_size, text_len)

            # Word boundary preservation:
            # If the chunk boundary falls in the middle of a word, backtrack to the nearest
            # whitespace character within a 20% margin of the boundary so words remain intact
            if end_idx < text_len:
                last_space = cleaned.rfind(" ", current_idx, end_idx)
                if last_space != -1 and (end_idx - last_space) < (self.chunk_size * 0.2):
                    end_idx = last_space

            # Extract the slice and store non-empty text
            chunk_str = cleaned[current_idx:end_idx].strip()
            if chunk_str:
                chunks.append(chunk_str)

            # Stop when the end of the document has been reached
            if end_idx >= text_len:
                break

            # Advance the starting index by step size, ensuring forward progress
            current_idx += step
            if current_idx >= end_idx:
                current_idx = end_idx

        return chunks
