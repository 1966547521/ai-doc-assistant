"""Tests for TranslationEngine functionality."""
import pytest
from unittest.mock import Mock, patch
from src.translation_engine import TranslationEngine


class TestTranslationEngine:
    """Test cases for TranslationEngine."""
    
    @pytest.fixture
    def translation_engine(self):
        """Create a TranslationEngine with mocked LLM."""
        with patch('src.translation_engine.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_llm.invoke.return_value.content = "Hello, this is a translated text."
            mock_get_llm.return_value = mock_llm
            
            engine = TranslationEngine()
            engine.llm = mock_llm
            return engine, mock_llm
    
    def test_translate_basic(self, translation_engine):
        """Test basic translation."""
        engine, mock_llm = translation_engine
        mock_llm.invoke.return_value.content = "Hello, this is a translated text."
        
        result = engine.translate("你好，这是一段中文文本。", target_lang="en")
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_translate_with_source_lang(self, translation_engine):
        """Test translation with explicit source language."""
        engine, mock_llm = translation_engine
        mock_llm.invoke.return_value.content = "Bonjour le monde"
        
        result = engine.translate("Hello world", target_lang="fr", source_lang="en")
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_translate_chunked(self, translation_engine):
        """Test chunked translation of long text."""
        engine, mock_llm = translation_engine
        mock_llm.invoke.return_value.content = "Translated chunk."
        
        result = engine.translate_chunked("A" * 3000, target_lang="en", chunk_size=1000)
        
        assert isinstance(result, str)
    
    def test_get_language_options(self, translation_engine):
        """Test getting language options for UI."""
        engine, _ = translation_engine
        
        options = engine.get_language_options()
        
        assert isinstance(options, list)
        assert len(options) > 0
        assert ("zh", "中文") in options
        assert ("en", "English") in options
    
    def test_get_lang_name_valid(self):
        """Test getting language name for valid code."""
        name = TranslationEngine._get_lang_name("zh")
        assert name == "中文"
        
        name = TranslationEngine._get_lang_name("en")
        assert name == "English"
    
    def test_get_lang_name_invalid(self):
        """Test getting language name for invalid code."""
        name = TranslationEngine._get_lang_name("invalid_code")
        assert name == "invalid_code"
    
    def test_stream_translate(self, translation_engine):
        """Test streaming translation."""
        engine, mock_llm = translation_engine
        
        def mock_stream(prompt):
            chunks = [Mock(content="Hello"), Mock(content=" world")]
            for chunk in chunks:
                yield chunk
        
        mock_llm.stream = mock_stream
        
        result_chunks = list(engine.stream_translate("你好世界", target_lang="en"))
        
        assert len(result_chunks) == 2
        assert result_chunks[0] == "Hello"
        assert result_chunks[1] == " world"
    
    def test_detect_language(self, translation_engine):
        """Test language detection."""
        engine, mock_llm = translation_engine
        mock_llm.invoke.return_value.content = "中文"
        
        result = engine.detect_language("你好，这是一段中文文本。")
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_supported_languages(self):
        """Test that supported languages list is correct."""
        langs = TranslationEngine.SUPPORTED_LANGUAGES
        
        assert len(langs) >= 8
        codes = [lang["code"] for lang in langs]
        assert "zh" in codes
        assert "en" in codes
        assert "ja" in codes
