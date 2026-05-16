"""Tests for utils functions."""
import os
import pytest
from unittest.mock import patch, Mock
import src.utils as utils


class TestUtils:
    """Test cases for utility functions."""
    
    def test_get_env_from_env_values(self):
        """Test getting environment value from .env file."""
        with patch.dict('src.utils._env_values', {'TEST_KEY': 'test_value'}, clear=True):
            result = utils._get_env('TEST_KEY')
            assert result == 'test_value'
    
    def test_get_env_from_system(self):
        """Test getting environment value from system env."""
        with patch.dict('src.utils._env_values', {}, clear=True):
            os.environ['TEST_SYSTEM_KEY'] = 'system_value'
            result = utils._get_env('TEST_SYSTEM_KEY')
            assert result == 'system_value'
            del os.environ['TEST_SYSTEM_KEY']
    
    def test_get_env_default(self):
        """Test getting default value."""
        with patch.dict('src.utils._env_values', {}, clear=True):
            result = utils._get_env('NONEXISTENT_KEY', 'default_val')
            assert result == 'default_val'
    
    def test_get_api_key(self):
        """Test getting API key from environment."""
        os.environ['TEST_API_KEY'] = 'test_api_key_value'
        result = utils._get_api_key('TEST_API_KEY')
        assert result == 'test_api_key_value'
        del os.environ['TEST_API_KEY']
    
    def test_get_api_key_not_found(self):
        """Test getting API key when not set."""
        result = utils._get_api_key('NONEXISTENT_API_KEY')
        assert result == ''
    
    def test_is_ollama_running_mock(self):
        """Test checking if Ollama is running."""
        with patch('src.utils.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            result = utils.is_ollama_running()
            assert result is True
    
    def test_is_ollama_not_running_mock(self):
        """Test checking if Ollama is not running."""
        with patch('src.utils.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_get.return_value = mock_response
            
            result = utils.is_ollama_running()
            assert result is False
    
    def test_is_ollama_connection_error(self):
        """Test Ollama check with connection error."""
        with patch('src.utils.requests.get', side_effect=ConnectionError):
            result = utils.is_ollama_running()
            assert result is False
    
    def test_get_available_ollama_models(self):
        """Test getting available Ollama models."""
        with patch('src.utils.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"name": "llama3.2"},
                    {"name": "qwen2.5"},
                    {"name": "nomic-embed-text"},
                ]
            }
            mock_get.return_value = mock_response
            
            models = utils.get_available_ollama_models()
            assert "llama3.2" in models
            assert "qwen2.5" in models
            assert "nomic-embed-text" in models
    
    def test_get_available_ollama_models_error(self):
        """Test getting Ollama models with error."""
        with patch('src.utils.requests.get', side_effect=ConnectionError):
            models = utils.get_available_ollama_models()
            assert models == []
    
    def test_get_default_ollama_embedding_model(self):
        """Test getting default embedding model."""
        with patch('src.utils.get_available_ollama_models', return_value=["nomic-embed-text", "other-model"]):
            model = utils.get_default_ollama_embedding_model()
            assert model == "nomic-embed-text"
    
    def test_get_default_ollama_embedding_model_fallback(self):
        """Test getting embedding model fallback."""
        with patch('src.utils.get_available_ollama_models', return_value=["unknown-model"]):
            model = utils.get_default_ollama_embedding_model()
            assert model == "unknown-model"
    
    def test_get_default_ollama_llm_model_qwen(self):
        """Test getting LLM model with Qwen available."""
        models = ["qwen2.5", "llama3.2"]
        with patch('src.utils.get_available_ollama_models', return_value=models):
            model = utils.get_default_ollama_llm_model()
            assert "qwen" in model.lower()
    
    def test_get_default_ollama_llm_model_preferred(self):
        """Test getting LLM model with preferred list."""
        with patch('src.utils.get_available_ollama_models', return_value=["llama3.2"]):
            model = utils.get_default_ollama_llm_model()
            assert model == "llama3.2"
    
    def test_get_default_ollama_llm_model_fallback(self):
        """Test getting LLM model fallback."""
        with patch('src.utils.get_available_ollama_models', return_value=["unknown-llm"]):
            model = utils.get_default_ollama_llm_model()
            assert model == "unknown-llm"
    
    def test_fallback_embeddings(self):
        """Test FallbackEmbeddings class."""
        primary = Mock()
        primary.embed_documents.return_value = [[0.1, 0.2]]
        primary.embed_query.return_value = [0.1, 0.2]
        
        fallback = Mock()
        fallback.embed_documents.return_value = [[0.3, 0.4]]
        fallback.embed_query.return_value = [0.3, 0.4]
        
        fb = utils.FallbackEmbeddings(primary, fallback)
        
        result = fb.embed_documents(["test"])
        assert result == [[0.1, 0.2]]
        
        result = fb.embed_query("test query")
        assert result == [0.1, 0.2]
    
    def test_fallback_embeddings_fallback_triggered(self):
        """Test FallbackEmbeddings triggers fallback on error."""
        primary = Mock()
        primary.embed_documents.side_effect = RuntimeError("API error")
        
        with patch('src.utils.is_ollama_running', return_value=True):
            fallback = Mock()
            fallback.embed_documents.return_value = [[0.3, 0.4]]
            
            fb = utils.FallbackEmbeddings(primary, fallback)
            
            result = fb.embed_documents(["test"])
            assert result == [[0.3, 0.4]]
            assert fb._use_fallback is True
    
    def test_get_llm_no_config(self):
        """Test get_llm with no configuration."""
        with patch.dict('src.utils._env_values', {}, clear=True):
            with patch('src.utils._get_api_key', return_value=''):
                with patch('src.utils.is_ollama_running', return_value=False):
                    with pytest.raises(RuntimeError, match="No valid API key found"):
                        utils.get_llm()
    
    def test_get_llm_with_ollama(self):
        """Test get_llm with Ollama."""
        with patch.dict('src.utils._env_values', {}, clear=True):
            with patch('src.utils._get_api_key', return_value=''):
                with patch('src.utils.is_ollama_running', return_value=True):
                    with patch('src.utils.get_default_ollama_llm_model', return_value='llama3.2'):
                        with patch('src.utils.ChatOpenAI') as mock_chat:
                            utils.get_llm()
                            mock_chat.assert_called_once()
