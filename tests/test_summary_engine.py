"""Tests for SummaryEngine functionality."""
import pytest
from unittest.mock import Mock, patch
from src.summary_engine import SummaryEngine


class TestSummaryEngine:
    """Test cases for SummaryEngine."""
    
    @pytest.fixture
    def summary_engine(self):
        """Create a SummaryEngine with mocked LLM."""
        with patch('src.summary_engine.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_llm.invoke.return_value.content = "This is a summary."
            mock_get_llm.return_value = mock_llm
            
            engine = SummaryEngine()
            engine.llm = mock_llm
            return engine, mock_llm
    
    def test_generate_summary_short(self, summary_engine):
        """Test generating a short summary."""
        engine, mock_llm = summary_engine
        mock_llm.invoke.return_value.content = "Short summary of the document."
        
        result = engine.generate_summary("This is a test document.", length="short")
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_generate_summary_detailed(self, summary_engine):
        """Test generating a detailed summary."""
        engine, mock_llm = summary_engine
        mock_llm.invoke.return_value.content = "Detailed summary."
        
        result = engine.generate_summary("Test document content.", length="detailed")
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_generate_summary_comprehensive(self, summary_engine):
        """Test generating a comprehensive summary."""
        engine, mock_llm = summary_engine
        mock_llm.invoke.return_value.content = "Comprehensive summary."
        
        result = engine.generate_summary("Test content.", length="comprehensive")
        
        assert isinstance(result, str)
    
    def test_generate_bullet_summary(self, summary_engine):
        """Test generating bullet-point summary."""
        engine, mock_llm = summary_engine
        mock_llm.invoke.return_value.content = "- Point 1\n- Point 2"
        
        result = engine.generate_bullet_summary("Document content.")
        
        assert isinstance(result, str)
        assert "Point" in result
    
    def test_generate_executive_summary(self, summary_engine):
        """Test generating executive summary."""
        engine, mock_llm = summary_engine
        mock_llm.invoke.return_value.content = "Executive summary content."
        
        result = engine.generate_executive_summary("Document.")
        
        assert isinstance(result, str)
    
    def test_generate_summary_with_questions(self, summary_engine):
        """Test generating summary with Q&A format."""
        engine, mock_llm = summary_engine
        mock_llm.invoke.return_value.content = "Q1: Answer1\nQ2: Answer2"
        
        result = engine.generate_summary_with_questions("Document.")
        
        assert isinstance(result, str)
    
    def test_stream_summary(self, summary_engine):
        """Test streaming summary generation."""
        engine, mock_llm = summary_engine
        
        def mock_stream(prompt):
            chunks = [Mock(content="Summary"), Mock(content=" content")]
            for chunk in chunks:
                yield chunk
        
        mock_llm.stream = mock_stream
        
        result_chunks = list(engine.stream_summary("Test document."))
        
        assert len(result_chunks) == 2
        assert result_chunks[0] == "Summary"
        assert result_chunks[1] == " content"
    
    def test_stream_bullet_summary(self, summary_engine):
        """Test streaming bullet summary."""
        engine, mock_llm = summary_engine
        
        def mock_stream(prompt):
            yield Mock(content="- Point")
        
        mock_llm.stream = mock_stream
        
        result_chunks = list(engine.stream_bullet_summary("Test."))
        
        assert len(result_chunks) == 1
        assert result_chunks[0] == "- Point"
    
    def test_stream_executive_summary(self, summary_engine):
        """Test streaming executive summary."""
        engine, mock_llm = summary_engine
        
        def mock_stream(prompt):
            yield Mock(content="Executive content")
        
        mock_llm.stream = mock_stream
        
        result_chunks = list(engine.stream_executive_summary("Test."))
        
        assert len(result_chunks) == 1
    
    def test_stream_summary_with_questions(self, summary_engine):
        """Test streaming Q&A summary."""
        engine, mock_llm = summary_engine
        
        def mock_stream(prompt):
            yield Mock(content="Q&A content")
        
        mock_llm.stream = mock_stream
        
        result_chunks = list(engine.stream_summary_with_questions("Test."))
        
        assert len(result_chunks) == 1
    
    def test_get_summary_prompt_default(self, summary_engine):
        """Test getting summary prompt falls back to default."""
        engine, _ = summary_engine
        
        result = engine._get_summary_prompt("nonexistent_prompt", "default text")
        
        assert result == "default text"
