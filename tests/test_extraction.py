import pytest
from app.core.exceptions import DocumentExtractionError, InvalidFileTypeError
from app.services.extraction.text_extractor import TextExtractor


def test_extract_txt_success():
    sample_text = "Palm Mind AI builds enterprise conversational systems.\nSecond line."
    content = sample_text.encode("utf-8")
    extracted, ext = TextExtractor.extract_text("doc.txt", content)
    assert ext == "txt"
    assert "Palm Mind AI" in extracted
    assert "Second line." in extracted


def test_extract_unsupported_extension():
    content = b"fake binary data"
    with pytest.raises(InvalidFileTypeError):
        TextExtractor.extract_text("image.jpg", content)


def test_extract_empty_txt():
    content = b"   \n\n  "
    with pytest.raises(DocumentExtractionError):
        TextExtractor.extract_text("empty.txt", content)


def test_extract_pdf_success():
    # Read the generated sample PDF
    with open("sample_data/palm_mind_handbook.pdf", "rb") as f:
        pdf_bytes = f.read()

    extracted, ext = TextExtractor.extract_text("palm_mind_handbook.pdf", pdf_bytes)
    assert ext == "pdf"
    assert "Palm Mind AI" in extracted
    assert "Interview Scheduling Protocol" in extracted
