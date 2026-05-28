"""Tests for document processing with real sample files and edge cases."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from src.document_processor import DocumentProcessor, clean_text


SAMPLE_DIR = Path(__file__).parent / "sample_docs"


# ── PDF helper: create minimal valid PDF bytes ─────────────────────────

MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n"
    b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"5 0 obj<</Length 44>>stream\n"
    b"BT /F1 12 Tf 100 700 Td (Hello PDF) Tj ET\n"
    b"endstream\n"
    b"endobj\n"
    b"xref\n"
    b"0 6\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000058 00000 n \n"
    b"0000000115 00000 n \n"
    b"0000000266 00000 n \n"
    b"0000000352 00000 n \n"
    b"trailer<</Size 6/Root 1 0 R>>\n"
    b"startxref\n"
    b"449\n"
    b"%%EOF\n"
)


# ── clean_text ─────────────────────────────────────────────────────────

class TestCleanText:
    def test_empty_string(self):
        assert clean_text("") == ""

    def test_whitespace_only(self):
        assert clean_text("   \n\n  ") == ""

    def test_normal_text_preserved(self):
        assert "你好 world" in clean_text("你好 world")

    def test_removes_bom(self):
        assert clean_text("\ufeffHello") == "Hello"

    def test_removes_zero_width_chars(self):
        result = clean_text("\u200b\u200c\u200dHello\u200e\u200f")
        assert result == "Hello"

    def test_normalizes_multiple_newlines(self):
        result = clean_text("Line1\n\n\n\nLine2")
        assert "Line1" in result and "Line2" in result
        assert "\n" in result

    def test_normalizes_spaces(self):
        result = clean_text("Hello     World")
        assert result == "Hello World"

    def test_removes_surrogates(self):
        result = clean_text("Hello\U0010FFFFWorld")
        assert result == "Hello World"

    def test_utf8_invalid_replacement(self):
        result = clean_text("Hello\x80World")
        assert "Hello" in result
        assert "World" in result


# ── DocumentProcessor with sample files ────────────────────────────────

class TestDocumentProcessorFiles:
    @pytest.fixture
    def proc(self):
        return DocumentProcessor()

    def test_read_txt(self, proc):
        path = str(SAMPLE_DIR / "sample_zh.txt")
        text = proc.read_txt(path)
        assert "项目背景" in text
        assert "技术架构" in text
        assert len(text) > 100

    def test_read_markdown(self, proc):
        path = str(SAMPLE_DIR / "sample_zh.md")
        text = proc.read_markdown(path)
        assert "智能文档助手" in text
        assert "RAG" in text

    def test_read_pdf(self, proc, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(MINIMAL_PDF)
        text = proc.read_pdf(str(pdf_path))
        assert "Hello PDF" in text

    def test_read_docx(self, proc):
        path = str(SAMPLE_DIR / "sample_zh.docx")
        if not os.path.exists(path):
            pytest.skip("sample_zh.docx not found")
        text = proc.read_docx(path)
        assert len(text) > 50

    def test_read_xlsx(self, proc):
        path = str(SAMPLE_DIR / "sample_zh.xlsx")
        if not os.path.exists(path):
            pytest.skip("sample_zh.xlsx not found")
        text = proc.read_xlsx(path)
        assert "DocumentProcessor" in text

    def test_read_pptx(self, proc):
        path = str(SAMPLE_DIR / "sample_zh.pptx")
        if not os.path.exists(path):
            pytest.skip("sample_zh.pptx not found")
        text = proc.read_pptx(path)
        assert len(text) > 20


# ── read_document dispatch ─────────────────────────────────────────────

class TestReadDocument:
    @pytest.fixture
    def proc(self):
        return DocumentProcessor()

    def test_dispatch_txt(self, proc, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("Hello", encoding="utf-8")
        result = proc.read_document(str(p))
        assert "Hello" in result

    def test_dispatch_md(self, proc, tmp_path):
        p = tmp_path / "test.md"
        p.write_text("# Title", encoding="utf-8")
        result = proc.read_document(str(p))
        assert "Title" in result

    def test_dispatch_pdf(self, proc, tmp_path):
        p = tmp_path / "test.pdf"
        p.write_bytes(MINIMAL_PDF)
        result = proc.read_document(str(p))
        assert "Hello PDF" in result

    def test_dispatch_docx(self, proc):
        path = str(SAMPLE_DIR / "sample_zh.docx")
        if not os.path.exists(path):
            pytest.skip("sample_zh.docx not found")
        result = proc.read_document(path)
        assert len(result) > 50

    def test_dispatch_xlsx(self, proc):
        path = str(SAMPLE_DIR / "sample_zh.xlsx")
        if not os.path.exists(path):
            pytest.skip("sample_zh.xlsx not found")
        result = proc.read_document(path)
        assert "DocumentProcessor" in result

    def test_dispatch_pptx(self, proc):
        path = str(SAMPLE_DIR / "sample_zh.pptx")
        if not os.path.exists(path):
            pytest.skip("sample_zh.pptx not found")
        result = proc.read_document(path)
        assert len(result) > 20

    def test_unsupported_format_raises(self, proc, tmp_path):
        p = tmp_path / "test.xyz"
        p.write_text("data", encoding="utf-8")
        with pytest.raises(ValueError, match="不支持"):
            proc.read_document(str(p))


# ── process_document ───────────────────────────────────────────────────

class TestProcessDocument:
    @pytest.fixture
    def proc(self):
        return DocumentProcessor()

    def test_process_txt(self, proc):
        path = str(SAMPLE_DIR / "sample_zh.txt")
        result = proc.process_document(path)
        assert "text" in result
        assert "chunks" in result
        assert "format" in result
        assert "char_count" in result
        assert result["format"] == "TXT"
        assert result["char_count"] > 0
        assert result["chunk_count"] > 0

    def test_process_docx(self, proc):
        path = str(SAMPLE_DIR / "sample_zh.docx")
        if not os.path.exists(path):
            pytest.skip("sample_zh.docx not found")
        result = proc.process_document(path)
        assert result["format"] == "DOCX"
        assert result["chunk_count"] > 0


# ── split_text ─────────────────────────────────────────────────────────

class TestSplitText:
    @pytest.fixture
    def proc(self):
        return DocumentProcessor()

    def test_split_short_text(self, proc):
        chunks = proc.split_text("Short text")
        assert len(chunks) == 1
        assert chunks[0].page_content == "Short text"

    def test_split_long_text(self, proc):
        long_text = "word " * 5000
        chunks = proc.split_text(long_text)
        assert len(chunks) > 1
        for c in chunks:
            assert c.page_content, "Each chunk must have content"

    def test_split_chinese_text(self, proc):
        chinese = "这是测试文档。" * 2000
        chunks = proc.split_text(chinese)
        assert len(chunks) > 1
        assert any("测试文档" in c.page_content for c in chunks)


# ── ImportError branches ──────────────────────────────────────────────

class TestImportErrorBranches:
    """Test that missing optional libs raise proper error."""

    @pytest.fixture
    def proc(self):
        return DocumentProcessor()

    def test_docx_import_error(self, proc, tmp_path):
        fake = tmp_path / "test.docx"
        fake.write_text("fake")
        with patch("src.document_processor.DOCX_AVAILABLE", False):
            with pytest.raises(ImportError, match="python-docx"):
                proc.read_docx(str(fake))

    def test_xlsx_import_error(self, proc, tmp_path):
        fake = tmp_path / "test.xlsx"
        fake.write_text("fake")
        with patch("src.document_processor.XLSX_AVAILABLE", False):
            with pytest.raises(ImportError, match="openpyxl"):
                proc.read_xlsx(str(fake))

    def test_pptx_import_error(self, proc, tmp_path):
        fake = tmp_path / "test.pptx"
        fake.write_text("fake")
        with patch("src.document_processor.PPTX_AVAILABLE", False):
            with pytest.raises(ImportError, match="python-pptx"):
                proc.read_pptx(str(fake))


# ── get_supported_formats / _is_format_available ───────────────────────

class TestFormatInfo:
    @pytest.fixture
    def proc(self):
        return DocumentProcessor()

    def test_supported_formats_contains_all(self, proc):
        fmts = proc.get_supported_formats()
        assert ".pdf" in fmts
        assert ".docx" in fmts
        assert ".xlsx" in fmts
        assert ".pptx" in fmts
        assert ".txt" in fmts
        assert ".md" in fmts

    def test_txt_always_available(self, proc):
        assert proc._is_format_available(".txt") is True
        assert proc._is_format_available(".md") is True
        assert proc._is_format_available(".pdf") is True
