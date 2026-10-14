import io
import re
from typing import Tuple
from pypdf import PdfReader

from app.core.exceptions import DocumentExtractionError, InvalidFileTypeError
from app.core.logging import logger


class TextExtractor:
    """Extracts clean text from .txt and .pdf files."""

    @staticmethod
    def extract_text(filename: str, content: bytes) -> Tuple[str, str]:
        # File extension routing:
        # Detect whether the file is a PDF or TXT based on the file extension
        # and delegate to the appropriate specialized extractor
        lower_name = filename.lower()

        if lower_name.endswith(".txt"):
            return TextExtractor._extract_from_txt(content), "txt"
        elif lower_name.endswith(".pdf"):
            return TextExtractor._extract_from_pdf(content), "pdf"
        else:
            raise InvalidFileTypeError(
                f"Unsupported file format for '{filename}'. Only .txt and .pdf are allowed."
            )

    @staticmethod
    def _extract_from_txt(content: bytes) -> str:
        """Text file decoding with encoding fallback:
        Uploaded text files can come with various character encodings (UTF-8, UTF-8 with BOM,
        or legacy Windows/Latin-1 encodings). We try the most common encodings sequentially
        to ensure broad compatibility without corrupting special characters.
        """
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                text = content.decode(encoding)
                cleaned = TextExtractor._normalize_text(text)

                # Ensure the file is not empty after removing whitespace
                if not cleaned:
                    raise DocumentExtractionError("The uploaded .txt file contains no readable text.")

                return cleaned
            except UnicodeDecodeError:
                continue

        raise DocumentExtractionError(
            "Failed to decode text file. Ensure it is encoded in UTF-8 or Latin-1."
        )

    @staticmethod
    def _extract_from_pdf(content: bytes) -> str:
        """PDF text extraction:
        Loads raw PDF bytes into an in-memory stream using pypdf, iterates through each page,
        and extracts readable text. Scanned documents without OCR text layers will result in
        an explicit, actionable error message.
        """
        try:
            # Read bytes into an in-memory stream for pypdf
            stream = io.BytesIO(content)
            reader = PdfReader(stream)
            extracted_pages = []

            # Extract text from every individual page in the document
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    extracted_pages.append(page_text.strip())

            # Detect scanned or image-only documents that lack embedded text
            if not extracted_pages:
                raise DocumentExtractionError(
                    "No extractable text was found in the PDF. Scanned or image-only PDFs are not supported."
                )

            # Combine page texts separated by double newlines for paragraph preservation
            full_text = "\n\n".join(extracted_pages)
            return TextExtractor._normalize_text(full_text)

        except DocumentExtractionError:
            raise
        except Exception as exc:
            logger.error(f"Error parsing PDF content: {exc}", exc_info=True)
            raise DocumentExtractionError(f"Failed to extract text from PDF: {str(exc)}") from exc

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Whitespace and newline normalization:
        Standardizes carriage returns to Unix newlines (\n), collapses runs of spaces
        or tabs into a single space, and limits multiple consecutive blank lines.
        """
        # Convert Windows and legacy Mac newlines to standard Unix newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse horizontal whitespace while preserving newline structure
        text = re.sub(r"[ \t]+", " ", text)

        # Consolidate excessive empty lines down to two newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
