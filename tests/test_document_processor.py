import pytest
import tempfile
import os
from src.document_processor import DocumentProcessor

class TestDocumentProcessor:
    def test_read_txt(self):
        processor = DocumentProcessor()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hello World\nThis is a test.")
            temp_path = f.name
        
        text = processor.read_txt(temp_path)
        assert "Hello World" in text
        assert "This is a test" in text
        os.unlink(temp_path)

    def test_split_text(self):
        processor = DocumentProcessor()
        text = "Hello " * 500
        documents = processor.split_text(text)
        assert len(documents) > 0
        assert all(len(doc.page_content) <= 1000 for doc in documents)

    def test_unsupported_format(self):
        processor = DocumentProcessor()
        with pytest.raises(ValueError):
            processor.read_document("test.xyz")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
