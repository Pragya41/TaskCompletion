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
        """Routes file to correct parser based on extension."""
        lower_name = filename.lower()

        # Check file extension
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
        """Decode .txt file bytes using common encodings."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

        # Try each encoding until one works
        for encoding in encodings:
            try:
                text = content.decode(encoding)
                cleaned = TextExtractor._normalize_text(text)

                # Make sure the file isn't empty
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
        """Extract text from PDF pages using pypdf."""
        try:
            # Read bytes into PdfReader
            stream = io.BytesIO(content)
            reader = PdfReader(stream)
            extracted_pages = []

            # Loop through each page and get text
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    extracted_pages.append(page_text.strip())

            # Error if no text found (e.g. scanned images)
            if not extracted_pages:
                raise DocumentExtractionError(
                    "No extractable text was found in the PDF. Scanned or image-only PDFs are not supported."
                )

            # Combine pages into a single string
            full_text = "\n\n".join(extracted_pages)
            return TextExtractor._normalize_text(full_text)

        except DocumentExtractionError:
            raise
        except Exception as exc:
            logger.error(f"Error parsing PDF content: {exc}", exc_info=True)
            raise DocumentExtractionError(f"Failed to extract text from PDF: {str(exc)}") from exc

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Clean up extra spaces, tabs, and line breaks."""
        # Normalize newlines
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Replace multiple spaces with a single space
        text = re.sub(r"[ \t]+", " ", text)

        # Remove extra blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
