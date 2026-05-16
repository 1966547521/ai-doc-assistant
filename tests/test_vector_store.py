"""Tests for VectorStoreManager functionality."""
import os
import shutil
import tempfile
import pytest
from langchain_core.documents import Document
from src.vector_store import VectorStoreManager


class TestVectorStoreManager:
    """Test cases for VectorStoreManager with incremental updates."""
    
    @pytest.fixture
    def temp_vectordb_dir(self):
        """Create a temporary directory for vector DB testing."""
        temp_dir = tempfile.mkdtemp(prefix="test_vectordb_")
        yield temp_dir
        # Cleanup with retry for Windows file locking issues
        if os.path.exists(temp_dir):
            import time
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(temp_dir)
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                    else:
                        # Skip cleanup if still locked
                        pass
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content="This is the first test document about AI and machine learning.",
                metadata={"source": "doc1"}
            ),
            Document(
                page_content="This is the second test document about natural language processing.",
                metadata={"source": "doc2"}
            ),
        ]
    
    @pytest.fixture
    def duplicate_documents(self):
        """Create duplicate documents for testing deduplication."""
        return [
            Document(
                page_content="This is the first test document about AI and machine learning.",
                metadata={"source": "doc1"}
            ),
            Document(
                page_content="This is the first test document about AI and machine learning.",
                metadata={"source": "doc1"}
            ),
        ]
    
    def test_content_hash_computation(self, temp_vectordb_dir):
        """Test content hash computation for deduplication."""
        vs_manager = VectorStoreManager(persist_directory=temp_vectordb_dir)
        content = "This is a test document."
        
        hash1 = vs_manager._compute_content_hash(content)
        hash2 = vs_manager._compute_content_hash(content)
        # Same content should have same hash
        assert hash1 == hash2
        
        # Different content should have different hash
        hash3 = vs_manager._compute_content_hash("Different content")
        assert hash1 != hash3
    
    def test_add_documents_incremental(self, temp_vectordb_dir, sample_documents):
        """Test adding documents with incremental update."""
        vs_manager = VectorStoreManager(persist_directory=temp_vectordb_dir)
        
        # Initial add
        result = vs_manager.add_documents(sample_documents, incremental=True)
        assert result["added"] == 2
        assert result["skipped"] == 0
        assert result["total"] == 2
        
        # Add again with incremental - should skip duplicates
        result2 = vs_manager.add_documents(sample_documents, incremental=True)
        assert result2["added"] == 0
        assert result2["skipped"] == 2
        assert result2["total"] == 2
    
    def test_add_documents_no_incremental(self, temp_vectordb_dir, sample_documents):
        """Test adding documents without incremental update."""
        vs_manager = VectorStoreManager(persist_directory=temp_vectordb_dir)
        
        # Initial add
        vs_manager.add_documents(sample_documents, incremental=True)
        
        # Add again without incremental - should add everything
        result = vs_manager.add_documents(sample_documents, incremental=False)
        assert result["added"] == 2
        assert result["skipped"] == 0
    
    def test_document_count(self, temp_vectordb_dir, sample_documents):
        """Test document count functionality."""
        vs_manager = VectorStoreManager(persist_directory=temp_vectordb_dir)
        
        assert vs_manager.get_document_count() == 0
        
        vs_manager.add_documents(sample_documents, incremental=True)
        assert vs_manager.get_document_count() == 2
    
    def test_clear_store(self, temp_vectordb_dir, sample_documents):
        """Test clearing the vector store."""
        vs_manager = VectorStoreManager(persist_directory=temp_vectordb_dir)
        vs_manager.add_documents(sample_documents, incremental=True)
        assert vs_manager.get_document_count() == 2
        
        vs_manager.clear_store()
        assert vs_manager.get_document_count() == 0
    
    def test_similarity_search(self, temp_vectordb_dir, sample_documents):
        """Test similarity search functionality."""
        vs_manager = VectorStoreManager(persist_directory=temp_vectordb_dir)
        vs_manager.add_documents(sample_documents, incremental=True)
        
        results = vs_manager.similarity_search("AI and machine learning", k=1)
        assert len(results) == 1
        assert "AI" in results[0].page_content
    
    def test_get_store(self, temp_vectordb_dir):
        """Test getting the vector store instance."""
        vs_manager = VectorStoreManager(persist_directory=temp_vectordb_dir)
        store = vs_manager.get_store()
        assert store is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
