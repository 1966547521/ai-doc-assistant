"""Tests for keyword_extractor module."""
import pytest
from unittest.mock import Mock, patch

from src.keyword_extractor import KeywordExtractor


class TestKeywordExtractor:
    """Test cases for KeywordExtractor class."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        mock = Mock()
        # 第一次调用提取关键词，第二次调用验证关键词
        mock.invoke.side_effect = [
            Mock(content="Python, Java, MySQL, LangChain, Streamlit"),
            Mock(content="Python, Java, MySQL")  # 验证后的结果
        ]
        mock.stream.return_value = [Mock(content="Python, "), Mock(content="Java, "), Mock(content="MySQL")]
        return mock
    
    @pytest.fixture
    def extractor(self, mock_llm):
        """Create a KeywordExtractor with mocked LLM."""
        with patch('src.keyword_extractor.get_llm', return_value=mock_llm):
            extractor = KeywordExtractor()
        return extractor
    
    def test_init(self):
        """Test initialization."""
        with patch('src.keyword_extractor.get_llm'):
            extractor = KeywordExtractor()
            assert extractor is not None
    
    def test_extract_key_terms(self, extractor, mock_llm):
        """Test keyword extraction with LLM validation."""
        text = "This is a test document about Python programming and machine learning."
        
        result = extractor.extract_key_terms(text)
        
        assert isinstance(result, list)
        assert len(result) > 0
        # 现在调用两次：提取 + 验证
        assert mock_llm.invoke.call_count == 2
    
    def test_extract_key_terms_without_validation(self, extractor, mock_llm):
        """Test keyword extraction without LLM validation."""
        text = "This is a test document about Python programming and machine learning."
        
        result = extractor.extract_key_terms(text, validate=False)
        
        assert isinstance(result, list)
        assert len(result) > 0
        # 不验证时只调用一次
        mock_llm.invoke.assert_called_once()
    
    def test_extract_key_terms_empty_text(self, extractor, mock_llm):
        """Test extraction with empty text."""
        result = extractor.extract_key_terms("")
        
        assert isinstance(result, list)
    
    def test_extract_actions(self, extractor, mock_llm):
        """Test action item extraction."""
        mock_llm.invoke.reset_mock()
        mock_llm.invoke.side_effect = [
            Mock(content="- Complete documentation\n- Review code\n- Fix bugs"),
            Mock(content="Complete documentation\nReview code")
        ]
        
        text = "We need to complete the documentation, review the code, and fix bugs."
        result = extractor.extract_actions(text)
        
        assert isinstance(result, list)
        assert len(result) >= 1
    
    def test_extract_actions_without_validation(self, extractor, mock_llm):
        """Test action item extraction without validation."""
        mock_llm.invoke.reset_mock()
        mock_llm.invoke.side_effect = None  # 清除side_effect
        mock_llm.invoke.return_value = Mock(content="- Complete documentation\n- Review code\n- Fix bugs")
        
        text = "We need to complete the documentation, review the code, and fix bugs."
        result = extractor.extract_actions(text, validate=False)
        
        assert isinstance(result, list)
        assert len(result) >= 1
        mock_llm.invoke.assert_called_once()
    
    def test_extract_topics(self, extractor, mock_llm):
        """Test topic extraction."""
        mock_llm.invoke.reset_mock()
        mock_llm.invoke.side_effect = [
            Mock(content="Python Programming\nMachine Learning\nData Analysis"),
            Mock(content="Python Programming\nMachine Learning")
        ]
        
        text = "This document discusses Python programming, machine learning, and data analysis."
        result = extractor.extract_topics(text)
        
        assert isinstance(result, list)
        assert len(result) >= 1
    
    def test_extract_topics_without_validation(self, extractor, mock_llm):
        """Test topic extraction without validation."""
        mock_llm.invoke.reset_mock()
        mock_llm.invoke.return_value = Mock(content="Python Programming\nMachine Learning\nData Analysis")
        
        text = "This document discusses Python programming, machine learning, and data analysis."
        result = extractor.extract_topics(text, validate=False)
        
        assert isinstance(result, list)
        assert len(result) >= 1
        mock_llm.invoke.assert_called_once()
    
    def test_stream_extract_key_terms(self, extractor, mock_llm):
        """Test streaming keyword extraction."""
        text = "Test document content"
        
        result = list(extractor.stream_extract_key_terms(text))
        
        assert isinstance(result, list)
        assert len(result) > 0
        mock_llm.stream.assert_called_once()
    
    def test_stream_extract_actions(self, extractor, mock_llm):
        """Test streaming action extraction."""
        mock_llm.stream.return_value = [Mock(content="- Task 1"), Mock(content="- Task 2")]
        
        result = list(extractor.stream_extract_actions("Test text"))
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_stream_extract_topics(self, extractor, mock_llm):
        """Test streaming topic extraction."""
        mock_llm.stream.return_value = [Mock(content="Topic 1"), Mock(content="Topic 2")]
        
        result = list(extractor.stream_extract_topics("Test text"))
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_stream_extract_key_terms_error(self, extractor, mock_llm):
        """Test error handling in streaming extraction."""
        mock_llm.stream.side_effect = ConnectionError("Network error")
        
        result = list(extractor.stream_extract_key_terms("Test text"))
        
        assert len(result) == 1
        assert "提取关键词时出现错误" in result[0]
    
    def test_get_prompt_default(self, extractor):
        """Test getting default prompt when file not found."""
        # This tests the _get_prompt method indirectly through extract_key_terms
        result = extractor.extract_key_terms("Test text", validate=False)
        assert isinstance(result, list)
    
    def test_max_terms_limit(self, extractor, mock_llm):
        """Test max_terms parameter limits results."""
        mock_llm.invoke.reset_mock()
        mock_llm.invoke.return_value = Mock(content="a, b, c, d, e, f, g, h, i, j, k")
        
        result = extractor.extract_key_terms("Test text", max_terms=5, validate=False)
        
        assert len(result) == 5
    
    def test_prompt_formatting_safety(self, extractor, mock_llm):
        """Test that prompt formatting handles missing placeholders."""
        # The extractor should handle prompts without {text} placeholder
        result = extractor.extract_key_terms("Test text", validate=False)
        assert isinstance(result, list)