"""Tests for QAEngine functionality."""
import pytest
from unittest.mock import Mock, patch
from src.qa_engine import QAEngine
from src.cache_manager import SemanticCacheManager


class TestQAEngine:
    """Test cases for QAEngine."""
    
    @pytest.fixture
    def qa_engine(self):
        """Create a QAEngine with mocked LLM."""
        with patch('src.qa_engine.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm
            
            engine = QAEngine(cache_manager=SemanticCacheManager())
            engine.llm = mock_llm
            return engine, mock_llm
    
    def test_answer_without_retriever(self, qa_engine):
        """Test answering without setting up retriever."""
        engine, _ = qa_engine
        
        result = engine.answer("What is this document about?")
        
        assert "answer" in result
        assert result["answer"] == "请先上传文档并建立索引"
        assert result["sources"] == []
    
    def test_set_retriever(self, qa_engine):
        """Test setting a retriever."""
        engine, _ = qa_engine
        
        mock_retriever = Mock()
        engine.set_retriever(mock_retriever)
        
        assert engine.retriever is not None
        assert engine.rag_chain is not None
    
    def test_set_context_snapshot(self, qa_engine):
        """Test setting context snapshot."""
        engine, _ = qa_engine
        
        engine.set_context_snapshot("test snapshot")
        
        assert engine._doc_snapshot == "test snapshot"
    
    def test_answer_with_rag_chain(self, qa_engine):
        """Test answering with RAG chain set up."""
        engine, mock_llm = qa_engine
        
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = [
            Mock(
                page_content="Relevant document content about AI.",
                metadata={"source_file": "演示.pdf", "page": 2, "chunk_id": "doc:0"},
            )
        ]
        
        class MockRagChain:
            def invoke(self, question):
                return "This is the answer based on documents."
        
        engine.set_retriever(mock_retriever)
        engine.rag_chain = MockRagChain()
        engine.cache_manager.clear_all()
        
        result = engine.answer("What is AI?")
        
        assert "answer" in result
        assert result["answer"] == "This is the answer based on documents."
        assert len(result["sources"]) > 0
        assert result["citations"] == [
            {"source_file": "演示.pdf", "page": 2, "chunk_id": "doc:0"}
        ]

    def test_answer_retrieves_once_and_uses_same_context_for_generation(self, qa_engine):
        engine, _ = qa_engine
        document = Mock(page_content="唯一检索片段", metadata={"page": 3})
        retriever = Mock()
        retriever.invoke.return_value = [document]

        class RecordingChain:
            payload = None

            def invoke(self, payload):
                self.payload = payload
                return "回答"

        chain = RecordingChain()
        engine.set_retriever(retriever)
        engine.rag_chain = chain
        engine.cache_manager.clear_all()

        result = engine.answer("问题")

        retriever.invoke.assert_called_once_with("问题")
        assert chain.payload["context"] == "唯一检索片段"
        assert result["sources"][0].startswith("唯一检索片段")
    
    def test_answer_with_chat_history(self, qa_engine):
        """Test answering with chat history."""
        engine, _ = qa_engine
        
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = [
            Mock(page_content="Document content.")
        ]
        
        class MockRagChain:
            def invoke(self, question):
                return "Answer with history context."
        
        engine.set_retriever(mock_retriever)
        engine.rag_chain = MockRagChain()
        engine.cache_manager.clear_all()
        
        result = engine.answer(
            "Follow up question?",
            chat_history="用户: 第一个问题\n助手: 第一个回答"
        )
        
        assert result["answer"] == "Answer with history context."
    
    def test_batch_answer(self, qa_engine):
        """Test batch answering."""
        engine, mock_llm = qa_engine
        
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []
        
        class MockRagChain:
            def invoke(self, question):
                return f"Answer to: {question}"
        
        engine.set_retriever(mock_retriever)
        engine.rag_chain = MockRagChain()
        engine.cache_manager.clear_all()
        
        results = engine.batch_answer(["Q1", "Q2", "Q3"])
        
        assert len(results) == 3
        assert "Q1" in results[0]["answer"]
        assert "Q2" in results[1]["answer"]
        assert "Q3" in results[2]["answer"]
    
    def test_stream_answer_without_retriever(self, qa_engine):
        """Test streaming answer without retriever."""
        engine, _ = qa_engine
        
        chunks = list(engine.stream_answer("test question"))
        
        assert len(chunks) == 1
        assert chunks[0] == "请先上传文档并建立索引"
    
    def test_stream_answer(self, qa_engine):
        """Test streaming answer with RAG chain."""
        engine, mock_llm = qa_engine
        
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []
        
        class MockRagChain:
            def stream(self, question):
                for text in ["Part 1 ", "Part 2"]:
                    yield text
        
        engine.set_retriever(mock_retriever)
        engine.rag_chain = MockRagChain()
        engine.cache_manager.clear_all()  # Clear cache to avoid stale hits
        
        chunks = list(engine.stream_answer("test question"))
        
        assert len(chunks) == 2
        assert chunks[0] == "Part 1 "
        assert chunks[1] == "Part 2"
    
    def test_get_sources(self, qa_engine):
        """Test getting sources."""
        engine, _ = qa_engine
        
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = [
            Mock(page_content="Source content one."),
            Mock(page_content="Source content two."),
        ]
        engine.set_retriever(mock_retriever)
        
        sources = engine.get_sources("test question")
        
        assert len(sources) == 2
        assert "Source content" in sources[0]
    
    def test_compute_context_hash(self, qa_engine):
        """Test computing context hash."""
        engine, _ = qa_engine
        
        engine.set_context_snapshot("test")
        hash1 = engine._compute_context_hash()
        hash2 = engine._compute_context_hash()
        
        assert hash1 == hash2  # Same snapshot produces same hash
        assert isinstance(hash1, str)
        assert len(hash1) > 0
