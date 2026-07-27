import pytest

from src.document_service import DocumentService


def test_document_service_rejects_unsupported_upload(tmp_path):
    path = tmp_path / "payload.exe"
    path.write_bytes(b"not a document")

    with pytest.raises(ValueError, match="仅支持 PDF 和 DOCX"):
        DocumentService().ingest_path(path, filename=path.name)


def test_document_id_is_content_based():
    assert DocumentService.document_id_for(b"same") == DocumentService.document_id_for(b"same")
    assert DocumentService.document_id_for(b"same") != DocumentService.document_id_for(b"other")
