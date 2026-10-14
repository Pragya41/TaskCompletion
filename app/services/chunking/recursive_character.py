from typing import List, Optional
from app.services.chunking.base import BaseChunker


class RecursiveCharacterChunker(BaseChunker):
    """Chunking Strategy 2: Splits hierarchically by paragraphs, lines, sentences, and words.

    Keeps natural paragraphs and sentences together within chunk_size.
    """

    # Priority of separators: paragraph -> line -> sentence -> word -> character
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, text: str) -> List[str]:
        cleaned = text.strip()
        if not cleaned:
            return []
        return self._split_text(cleaned, self.separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Hierarchical text decomposition:
        Attempts to split text using the coarsest available delimiter first (paragraphs),
        recursing down to smaller boundaries (sentences, words) only when individual pieces
        exceed the maximum chunk size.
        """
        separator = separators[-1]
        new_separators: List[str] = []

        # Find the highest priority separator present in the text
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split using the selected separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        good_splits: List[str] = [s for s in splits if s]

        # Combine smaller splits into chunks that fit chunk_size with overlap
        return self._merge_splits(good_splits, separator, new_separators)

    def _merge_splits(
        self, splits: List[str], separator: str, next_separators: List[str]
    ) -> List[str]:
        """Greedy accumulator with overlap:
        Merges atomic text fragments into chunks that respect `chunk_size`.
        When a chunk boundary is reached, it retains trailing fragments within
        `chunk_overlap` to maintain semantic continuity between consecutive chunks.
        """
        docs: List[str] = []
        current_doc: List[str] = []
        total_len = 0

        for split in splits:
            split_len = len(split)

            # If a single split is still larger than chunk_size, recursively subdivide it
            # using the next delimiter in the priority hierarchy
            if split_len > self.chunk_size:
                if current_doc:
                    merged = separator.join(current_doc).strip()
                    if merged:
                        docs.append(merged)
                    current_doc = []
                    total_len = 0

                if next_separators:
                    sub_chunks = self._split_text(split, next_separators)
                    docs.extend(sub_chunks)
                else:
                    for start in range(0, split_len, self.chunk_size - self.chunk_overlap):
                        sub = split[start : start + self.chunk_size].strip()
                        if sub:
                            docs.append(sub)
                continue

            # If adding this piece exceeds chunk_size, save current chunk and start a new one
            sep_len = len(separator) if current_doc else 0
            if total_len + split_len + sep_len > self.chunk_size:
                if current_doc:
                    merged = separator.join(current_doc).strip()
                    if merged:
                        docs.append(merged)

                    # Carry forward trailing fragments to form the overlap for the next chunk
                    overlap_doc: List[str] = []
                    overlap_len = 0
                    for prev_split in reversed(current_doc):
                        p_sep = len(separator) if overlap_doc else 0
                        if overlap_len + len(prev_split) + p_sep <= self.chunk_overlap:
                            overlap_doc.insert(0, prev_split)
                            overlap_len += len(prev_split) + p_sep
                        else:
                            break

                    current_doc = overlap_doc
                    total_len = overlap_len

            # Add fragment to current accumulation buffer
            current_doc.append(split)
            total_len += split_len + (len(separator) if len(current_doc) > 1 else 0)

        # Flush any remaining text in the buffer
        if current_doc:
            merged = separator.join(current_doc).strip()
            if merged:
                docs.append(merged)

        return docs
