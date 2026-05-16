"""Question answering engine using RAG with custom prompts, semantic caching, and logging.

This module provides document-based question answering capabilities with caching,
LLM evaluation, and comprehensive logging support for monitoring and debugging.
"""

import hashlib
from typing import Dict, Iterator, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.cache_manager import SemanticCacheManager
from src.prompt_manager import prompt_manager
from src.utils import get_llm
from src.logger import get_logger

# Initialize module logger
logger = get_logger(__name__)


class QAEngine:
    """QA engine for document-based question answering with caching and LLM evaluation."""

    def __init__(self, cache_manager: Optional[SemanticCacheManager] = None, llm: Optional[BaseChatModel] = None):
        logger.info("Initializing QAEngine")
        self.llm = llm if llm is not None else get_llm()
        self.rag_chain = None
        self.retriever = None
        self._context_hash: Optional[str] = None
        self._cached_questions: List[str] = []
        self._doc_snapshot: str = ""
        self.cache_manager = cache_manager or SemanticCacheManager()
        self._enhancer = None
        logger.debug("QAEngine initialized successfully")

    def _get_enhancer(self):
        """延迟初始化LLM增强器"""
        if self._enhancer is None:
            from .llm_enhancer import LLMEnhancer
            self._enhancer = LLMEnhancer(self.llm)
        return self._enhancer

    def _compute_context_hash(self) -> str:
        """Compute a hash of the current document context"""
        if self._context_hash is None:
            # Generate a stable context identifier
            if hasattr(self, "_doc_snapshot"):
                doc_snapshot = self._doc_snapshot
            else:
                doc_snapshot = str(id(self.retriever)) if self.retriever else "empty"
            self._context_hash = hashlib.md5(doc_snapshot.encode()).hexdigest()
        return self._context_hash

    def set_context_snapshot(self, snapshot: str) -> None:
        """Set a snapshot of the current document state for caching"""
        self._doc_snapshot = snapshot
        self._context_hash = None  # Reset hash

    def set_retriever(self, retriever):
        """Set the document retriever for the QA chain.
        
        Args:
            retriever: Document retriever to use for context retrieval
        """
        logger.info("Setting retriever for QA chain")
        
        # Load prompt from file or use default
        template = prompt_manager.get_prompt(
            "qa",
            """你是一个智能助手，请根据以下上下文回答问题。
如果上下文中没有相关信息，请说"根据提供的文档，我无法找到相关信息"。

上下文：
{context}

问题：{question}

请提供详细、准确的回答：""",
        )

        prompt = ChatPromptTemplate.from_template(template)

        self.rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        self.retriever = retriever
        self._context_hash = None
        self._cached_questions = []
        
        logger.debug("QA chain configured successfully")

    def answer(self, question: str, chat_history: str = "", evaluate: bool = False) -> Dict[str, str | list | Dict]:
        """Answer a question based on the indexed documents with caching.
        
        Args:
            question: The question to answer
            chat_history: Previous chat history for context
            evaluate: Whether to evaluate answer quality with LLM
        
        Returns:
            Dictionary containing answer, sources, and evaluation
        """
        logger.info(f"Answering question (evaluate={evaluate})")
        logger.debug(f"Question length: {len(question)} chars, chat_history: {len(chat_history)} chars")

        if not self.rag_chain:
            logger.warning("QA chain not configured, returning error")
            return {"answer": "请先上传文档并建立索引", "sources": [], "evaluation": None}

        if not question.strip():
            logger.warning("Empty question provided")
            return {"answer": "请输入有效的问题", "sources": [], "evaluation": None}

        try:
            # Check cache first
            context_hash = self._compute_context_hash()
            cached_answer = self.cache_manager.get_qa(question, context_hash)
            if cached_answer:
                logger.debug("Found cached answer")
                sources = self.get_sources(question, chat_history)
                return {"answer": cached_answer, "sources": sources, "evaluation": None}

            full_question = question
            if chat_history:
                full_question = f"历史对话:\n{chat_history}\n\n当前问题:\n{question}"

            logger.debug("Invoking RAG chain for answer")
            answer = self.rag_chain.invoke(full_question)

            # Cache the result
            self.cache_manager.cache_qa(question, context_hash, answer)
            self._cached_questions.append(question)
            logger.debug("Answer cached successfully")

            sources = []
            context = ""
            if self.retriever:
                docs = self.retriever.invoke(full_question)
                sources = [doc.page_content[:100] + "..." for doc in docs]
                context = "\n".join([doc.page_content[:300] for doc in docs])

            # LLM评估回答质量
            evaluation = None
            if evaluate:
                logger.debug("Evaluating answer quality with LLM")
                enhancer = self._get_enhancer()
                evaluation = enhancer.evaluate_answer(question, answer, context)

            logger.info(f"Answer generated, length: {len(answer)}")
            return {"answer": answer, "sources": sources, "evaluation": evaluation}
        
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}", exc_info=True)
            return {"answer": f"回答问题时出现错误: {str(e)}", "sources": [], "evaluation": None}

    def batch_answer(self, questions: List[str], evaluate: bool = False) -> List[Dict[str, str | list | Dict]]:
        """Answer multiple questions in batch.
        
        Args:
            questions: List of questions to answer
            evaluate: Whether to evaluate answer quality with LLM
        
        Returns:
            List of answer dictionaries
        """
        logger.info(f"Batch answering {len(questions)} questions")
        
        results: List[Dict[str, str | list | Dict]] = []
        for i, question in enumerate(questions):
            logger.debug(f"Processing question {i+1}/{len(questions)}")
            results.append(self.answer(question, evaluate=evaluate))
        
        logger.info("Batch answering completed")
        return results

    def stream_answer(self, question: str, chat_history: str = "") -> Iterator[str]:
        """Stream answer to a question based on the indexed documents.
        
        Args:
            question: The question to answer
            chat_history: Previous chat history for context
        
        Yields:
            Streaming answer chunks
        """
        logger.info("Streaming answer")

        if not self.rag_chain:
            logger.warning("QA chain not configured")
            yield "请先上传文档并建立索引"
            return

        if not question.strip():
            logger.warning("Empty question provided for streaming")
            yield "请输入有效的问题"
            return

        try:
            # Check cache first
            context_hash = self._compute_context_hash()
            cached_answer = self.cache_manager.get_qa(question, context_hash)
            if cached_answer:
                logger.debug("Streaming cached answer")
                yield "📝 (来自缓存) "
                for char in cached_answer:
                    yield char
                return

            full_question = question
            if chat_history:
                full_question = f"历史对话:\n{chat_history}\n\n当前问题:\n{question}"

            logger.debug("Streaming RAG chain response")
            stream = self.rag_chain.stream(full_question)
            full_answer = []
            for chunk in stream:
                if isinstance(chunk, ChatGenerationChunk):
                    text = chunk.text
                    full_answer.append(text)
                    yield text
                else:
                    full_answer.append(chunk)
                    yield chunk

            # Cache the result after streaming completes
            self.cache_manager.cache_qa(question, context_hash, "".join(full_answer))
            self._cached_questions.append(question)
            logger.debug("Streamed answer cached successfully")
        
        except (ConnectionError, RuntimeError) as e:
            logger.error(f"Error streaming answer: {str(e)}", exc_info=True)
            yield f"回答过程中出现错误: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in stream_answer: {str(e)}", exc_info=True)
            yield f"回答过程中出现错误: {str(e)}"

    def get_sources(self, question: str, chat_history: str = "") -> List[str]:
        """Get source documents for a question."""
        if not self.retriever:
            return []

        full_question = question
        if chat_history:
            full_question = f"历史对话:\n{chat_history}\n\n当前问题:\n{question}"

        docs = self.retriever.invoke(full_question)
        return [doc.page_content[:100] + "..." for doc in docs]
