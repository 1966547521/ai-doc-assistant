"""Tests for DocumentComparer functionality."""
import pytest
from unittest.mock import patch
from src.document_comparer import DocumentComparer


class TestDocumentComparer:
    """Test cases for DocumentComparer."""
    
    @pytest.fixture
    def comparer(self):
        """Create a DocumentComparer instance."""
        return DocumentComparer()
    
    def test_compare_texts_basic(self, comparer):
        """Test basic text comparison."""
        text1 = "Hello World"
        text2 = "Hello World"
        
        result = comparer.compare_texts(text1, text2)
        
        assert "similarity" in result
        assert "added" in result
        assert "removed" in result
        assert "stats" in result
        assert result["similarity"] == 100.0  # Identical texts (percentage)
    
    def test_compare_texts_different(self, comparer):
        """Test comparing different texts."""
        text1 = "Hello World"
        text2 = "Hello Python"
        
        result = comparer.compare_texts(text1, text2)
        
        assert result["similarity"] < 100.0  # Different texts (percentage)
        assert "Hello World" in result["removed"]
        assert "Hello Python" in result["added"]
    
    def test_compare_texts_empty(self, comparer):
        """Test comparing empty texts."""
        text1 = ""
        text2 = "Hello World"
        
        result = comparer.compare_texts(text1, text2)
        
        assert len(result["added"]) > 0
        assert len(result["removed"]) == 0
    
    def test_compare_texts_lines(self, comparer):
        """Test comparing multi-line texts."""
        text1 = """Line 1
Line 2
Line 3"""
        text2 = """Line 1
Line 2 Modified
Line 4"""
        
        result = comparer.compare_texts(text1, text2)
        
        assert "Line 3" in result["removed"]
        assert "Line 2 Modified" in result["added"]
        assert "Line 4" in result["added"]
        assert "Line 1" in result["unchanged"]
    
    def test_calculate_similarity(self, comparer):
        """Test similarity calculation."""
        text1 = "The quick brown fox"
        text2 = "The quick brown fox"
        assert comparer.calculate_similarity(text1, text2) == 100.0  # Returns percentage
        
        text3 = "The lazy dog"
        similarity = comparer.calculate_similarity(text1, text3)
        assert 0 < similarity < 100.0  # Percentage range
    
    def test_generate_diff_summary(self, comparer):
        """Test generating diff summary."""
        text1 = "Hello World"
        text2 = "Hello Python"
        
        # Mock the LLM to avoid actual API calls
        with patch.object(comparer, 'llm') as mock_llm:
            mock_llm.invoke.return_value.content = "文档对比总结"
            summary = comparer.generate_diff_summary(text1, text2)
        
        assert isinstance(summary, str)
        assert len(summary) > 0