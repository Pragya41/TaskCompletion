import io
import re
from typing import Tuple
from pypdf import PdfReader

from app.core.exceptions import DocumentExtractionError, InvalidFileTypeError
from app.core.logging import logger


class TextExtractor:
    """Service for extracting raw text from uploaded .pdf and .txt files."""

    @staticmethod
    def extract_text(filename: str, content: bytes) -> Tuple[str, str]:
        """Extracts normalized text from the given file bytes.

        Args:
            filename: Name of the uploaded file.
            content: Raw byte content.

        Returns:
            Tuple of (extracted_text, file_extension)

        Raises:
            InvalidFileTypeError: If the file type is not supported.
            DocumentExtractionError: If text extraction fails or yields empty text.
        """
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
        """Decodes raw bytes into text using common encodings."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        for encoding in encodings:
            try:
                text = content.decode(encoding)
                cleaned = TextExtractor._normalize_text(text)
                if not cleaned:
                    raise DocumentExtractionError("The uploaded .txt file contains no readable text.")
                return cleaned
            except UnicodeDecodeError:
                continue

        raise DocumentExtractionError("Failed to decode text file. Ensure it is encoded in UTF-8 or Latin-1.")

    @staticmethod
    def _extract_from_pdf(content: bytes) -> str:
        """Extracts text page-by-page from a PDF document using pypdf."""
        try:
            stream = io.BytesIO(content)
            reader = PdfReader(stream)
            extracted_pages = []

            for page_index, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    extracted_pages.append(page_text.strip())

            if not extracted_pages:
                raise DocumentExtractionError(
                    "No extractable text was found in the PDF. Scanned or image-only PDFs are not supported."
                )

            full_text = "\n\n".join(extracted_pages)
            return TextExtractor._normalize_text(full_text)
        except DocumentExtractionError:
            raise
        except Exception as exc:
            logger.error(f"Error parsing PDF content: {exc}", exc_info=True)
            raise DocumentExtractionError(f"Failed to extract text from PDF: {str(exc)}") from exc

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Standardizes linebreaks, tabs, and excess whitespace."""
        # Replace Windows and old Mac line breaks
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Replace multiple horizontal spaces/tabs with single space
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse excessive newlines (more than 2 consecutive newlines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
