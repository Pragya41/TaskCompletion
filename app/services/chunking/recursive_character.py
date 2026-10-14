from typing import List, Optional
from app.services.chunking.base import BaseChunker


class RecursiveCharacterChunker(BaseChunker):
    """Hierarchical text splitter that splits text based on prioritized separators:

    ["\\n\\n", "\\n", ". ", " ", ""] to keep coherent paragraphs and sentences together.
    """

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
        final_chunks: List[str] = []

        # Find the best separator that appears in this text segment
        separator = separators[-1]
        new_separators: List[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split text by separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Merge splits into chunks under chunk_size with chunk_overlap
        good_splits: List[str] = []
        for s in splits:
            if s:
                good_splits.append(s)

        merged_chunks = self._merge_splits(good_splits, separator, new_separators)
        return merged_chunks

    def _merge_splits(
        self, splits: List[str], separator: str, next_separators: List[str]
    ) -> List[str]:
        docs: List[str] = []
        current_doc: List[str] = []
        total_len = 0

        for split in splits:
            split_len = len(split)

            # If an individual split is larger than chunk_size, recursively split it
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
                    # Character level slicing if no separators left
                    for start in range(0, split_len, self.chunk_size - self.chunk_overlap):
                        sub = split[start : start + self.chunk_size].strip()
                        if sub:
                            docs.append(sub)
                continue

            # Check if adding this split exceeds chunk_size
            sep_len = len(separator) if current_doc else 0
            if total_len + split_len + sep_len > self.chunk_size:
                if current_doc:
                    merged = separator.join(current_doc).strip()
                    if merged:
                        docs.append(merged)

                    # Keep overlap from the end of current_doc for the next chunk
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

            current_doc.append(split)
            total_len += split_len + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            merged = separator.join(current_doc).strip()
            if merged:
                docs.append(merged)

        return docs
