"""Tests for SemanticCacheManager functionality."""
import os
import shutil
import tempfile
import pytest
from src.cache_manager import SemanticCacheManager


class TestSemanticCacheManager:
    """Test cases for SemanticCacheManager."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        temp_dir = tempfile.mkdtemp(prefix="test_cache_")
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create a SemanticCacheManager instance for testing."""
        return SemanticCacheManager(
            cache_dir=temp_cache_dir,
            max_entries=10,
            default_ttl=3600
        )
    
    def test_basic_cache_operations(self, cache_manager):
        """Test basic set and get operations."""
        cache_manager.set("key1", "value1")
        result = cache_manager.get("key1")
        assert result == "value1"
    
    def test_cache_miss_returns_default(self, cache_manager):
        """Test that missing keys return default value."""
        assert cache_manager.get("nonexistent", "default") == "default"
    
    def test_cache_stats(self, cache_manager):
        """Test cache statistics calculation."""
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        
        stats = cache_manager.get_cache_stats()
        assert stats["total_entries"] == 2
        assert "total_size_human" in stats
    
    def test_clear_cache(self, cache_manager):
        """Test cache clearing functionality."""
        cache_manager.set("key1", "value1")
        assert cache_manager.get("key1") == "value1"
        
        cache_manager.clear()
        assert cache_manager.get("key1") is None
    
    def test_similarity_comparison(self, cache_manager):
        """Test text similarity comparison."""
        # Check that similar texts get high similarity
        similarity = cache_manager._compute_similarity(
            "Hello world, this is a test",
            "Hello world, this is another test"
        )
        assert similarity > 0.5
        
        # Check that different texts get low similarity
        dissimilarity = cache_manager._compute_similarity(
            "Hello world",
            "Goodbye universe"
        )
        assert dissimilarity < 0.7
    
    def test_document_summary_cache(self, cache_manager):
        """Test document summary caching."""
        doc_text = "This is a long document text for testing summary caching."
        summary = "This is a summary of the document."
        
        cache_manager.cache_document_summary(doc_text, summary)
        cached = cache_manager.get_document_summary(doc_text)
        assert cached == summary
    
    def test_qa_cache(self, cache_manager):
        """Test QA response caching."""
        question = "What is this?"
        context_hash = "context123"
        answer = "This is a test answer."
        
        cache_manager.cache_qa(question, context_hash, answer)
        cached = cache_manager.get_qa(question, context_hash)
        assert cached == answer
    
    def test_lru_eviction(self, temp_cache_dir):
        """Test LRU eviction when cache exceeds max entries."""
        cache = SemanticCacheManager(
            cache_dir=temp_cache_dir,
            max_entries=3,
            default_ttl=3600
        )
        
        # Add more entries than max
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")
        
        # Key1 should have been evicted
        assert cache.get("key1") is None
        # Other keys should exist
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None
    
    def test_format_size(self, cache_manager):
        """Test size formatting functionality."""
        size_str = cache_manager._format_size(1024)
        assert "KB" in size_str
        
        size_str = cache_manager._format_size(1024 * 1024)
        assert "MB" in size_str
    
    def test_clear_all(self, temp_cache_dir):
        """Test clear_all functionality."""
        cache = SemanticCacheManager(cache_dir=temp_cache_dir)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        cache.clear_all()
        assert cache.get("key1") is None
        assert cache.get("key2") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
