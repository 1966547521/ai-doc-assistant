"""Keyword extraction module using LLM with streaming support and logging.

This module provides keyword, action item, and topic extraction capabilities
with LLM validation for improved accuracy. All operations are logged for
monitoring and debugging purposes.
"""

import re
from typing import Iterator, List, Optional

from langchain_core.language_models import BaseChatModel

from src.prompt_manager import prompt_manager
from src.utils import get_llm
from src.logger import get_logger

# Initialize module logger
logger = get_logger(__name__)


class KeywordExtractor:
    """Extracts keywords and key terms from document content with LLM validation.

    Features:
        - Extracts keywords, action items, and topics
        - Validates extracted items using LLM for improved relevance
        - Supports streaming extraction for real-time results
        - Comprehensive logging for monitoring
    """

    MAX_TEXT_LENGTH = 5000
    VALIDATION_MULTIPLIER = 2
    MIN_ITEMS_FOR_KEYWORD_VALIDATION = 3
    MIN_ITEMS_FOR_ACTION_VALIDATION = 2
    MIN_ITEMS_FOR_TOPIC_VALIDATION = 2

    def __init__(self, llm: Optional[BaseChatModel] = None):
        """Initialize the keyword extractor."""
        logger.info("Initializing KeywordExtractor")
        self.llm = llm if llm is not None else get_llm()
        self._enhancer = None
        logger.debug("KeywordExtractor initialized successfully")

    def _get_enhancer(self):
        """延迟初始化LLM增强器（延迟加载以优化性能）"""
        if self._enhancer is None:
            logger.debug("Creating LLMEnhancer instance")
            from .llm_enhancer import LLMEnhancer
            self._enhancer = LLMEnhancer(self.llm)
            logger.debug("LLMEnhancer instance created")
        return self._enhancer

    def _get_prompt(self, prompt_name: str, default: str) -> str:
        """Get prompt from file or use default.

        Args:
            prompt_name: Name of the prompt to retrieve
            default: Default prompt text if not found

        Returns:
            The prompt text
        """
        result = prompt_manager.get_prompt(prompt_name, default)
        if result is None:
            logger.debug(f"Prompt '{prompt_name}' not found, using default")
            return default
        logger.debug(f"Loaded prompt '{prompt_name}'")
        return result

    def _prepare_prompt(self, prompt_name: str, default: str, text: str) -> str:
        prompt = self._get_prompt(prompt_name, default)
        truncated_text = text[:self.MAX_TEXT_LENGTH]
        if "{text}" in prompt:
            return prompt.format(text=truncated_text)
        return prompt + "\n\n【待提取文本】\n" + truncated_text

    def extract_key_terms(self, text: str, max_terms: int = 10, validate: bool = True) -> List[str]:
        """Extract key terms and important keywords from text.

        Args:
            text: The input text to extract keywords from
            max_terms: Maximum number of keywords to return (default: 10)
            validate: Whether to validate keywords with LLM for relevance (default: True)

        Returns:
            List of extracted keywords, sorted by relevance
        """
        logger.info(f"Extracting key terms from text (max_terms={max_terms}, validate={validate})")
        logger.debug(f"Input text length: {len(text)} characters")

        if not text.strip():
            logger.warning("Empty text provided for keyword extraction")
            return []

        prompt = self._prepare_prompt(
            "keyword_extract",
            """请从以下文本中提取关键词：

{text}

请以逗号分隔输出关键词，不需要解释。""",
            text,
        )

        try:
            logger.debug("Invoking LLM for keyword extraction")
            response = self.llm.invoke(prompt)
            terms = [t.strip() for t in response.content.split(",")]
            terms = [t for t in terms if t][:max_terms * self.VALIDATION_MULTIPLIER]
            logger.debug(f"Extracted {len(terms)} initial keywords")

            if validate and len(terms) > self.MIN_ITEMS_FOR_KEYWORD_VALIDATION:
                logger.debug("Validating keywords with LLMEnhancer")
                enhancer = self._get_enhancer()
                terms = enhancer.validate_keywords(text, terms)
                logger.debug(f"After validation: {len(terms)} keywords remaining")

            result = terms[:max_terms]
            logger.info(f"Keyword extraction completed, found {len(result)} keywords")
            return result

        except Exception as e:
            logger.error(f"Error extracting key terms: {str(e)}", exc_info=True)
            return []

    def extract_actions(self, text: str, validate: bool = True) -> List[str]:
        """Extract action items or tasks from document.

        Args:
            text: The input text to extract action items from
            validate: Whether to validate actions with LLM for relevance (default: True)

        Returns:
            List of extracted action items
        """
        logger.info(f"Extracting action items (validate={validate})")
        logger.debug(f"Input text length: {len(text)} characters")

        if not text.strip():
            logger.warning("Empty text provided for action item extraction")
            return []

        prompt = self._prepare_prompt(
            "action_extract",
            """请从以下文本中提取需要执行的行动项或任务：

{text}

请列出所有需要执行的任务或行动项，用项目符号列表形式输出。""",
            text,
        )

        try:
            logger.debug("Invoking LLM for action extraction")
            response = self.llm.invoke(prompt)
            lines = response.content.split("\n")
            actions = []
            for line in lines:
                line = line.strip()
                if line and (
                    line.startswith("-") or line.startswith("*") or line[0].isdigit()
                ):
                    actions.append(re.sub(r"^[-*]+\s*|\d+\.\s*", "", line))
            logger.debug(f"Extracted {len(actions)} initial action items")

            if validate and len(actions) > self.MIN_ITEMS_FOR_ACTION_VALIDATION:
                logger.debug("Validating action items with LLMEnhancer")
                enhancer = self._get_enhancer()
                actions = enhancer.validate_actions(text, actions)
                logger.debug(f"After validation: {len(actions)} action items remaining")

            logger.info(f"Action extraction completed, found {len(actions)} items")
            return actions

        except Exception as e:
            logger.error(f"Error extracting action items: {str(e)}", exc_info=True)
            return []

    def extract_topics(self, text: str, max_topics: int = 5, validate: bool = True) -> List[str]:
        """Extract main topics from document.

        Args:
            text: The input text to extract topics from
            max_topics: Maximum number of topics to return (default: 5)
            validate: Whether to validate topics with LLM for relevance (default: True)

        Returns:
            List of extracted topics
        """
        logger.info(f"Extracting topics (max_topics={max_topics}, validate={validate})")
        logger.debug(f"Input text length: {len(text)} characters")

        if not text.strip():
            logger.warning("Empty text provided for topic extraction")
            return []

        prompt = self._prepare_prompt(
            "topic_extract",
            """请从以下文本中提取主要主题：

{text}

请列出文档讨论的主要主题，每个主题一行，不需要解释。""",
            text,
        )

        try:
            logger.debug("Invoking LLM for topic extraction")
            response = self.llm.invoke(prompt)
            topics = [t.strip() for t in response.content.split("\n")]
            topics = [t for t in topics if t][:max_topics * self.VALIDATION_MULTIPLIER]
            logger.debug(f"Extracted {len(topics)} initial topics")

            if validate and len(topics) > self.MIN_ITEMS_FOR_TOPIC_VALIDATION:
                logger.debug("Validating topics with LLMEnhancer")
                enhancer = self._get_enhancer()
                topics = enhancer.validate_topics(text, topics)
                logger.debug(f"After validation: {len(topics)} topics remaining")

            result = topics[:max_topics]
            logger.info(f"Topic extraction completed, found {len(result)} topics")
            return result

        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}", exc_info=True)
            return []

    def stream_extract_key_terms(self, text: str, max_terms: int = 10) -> Iterator[str]:
        """Stream key terms extraction for real-time results.

        Args:
            text: The input text to extract keywords from
            max_terms: Maximum number of keywords to return (default: 10)

        Yields:
            Streaming chunks of keyword extraction results
        """
        logger.info(f"Streaming key term extraction (max_terms={max_terms})")

        if not text.strip():
            logger.warning("Empty text provided for streaming keyword extraction")
            yield "未提供有效文本"
            return

        prompt = self._prepare_prompt(
            "keyword_extract",
            """请从以下文本中提取最多10个关键词或关键术语：

{text}

请以逗号分隔输出关键词，不需要解释。""",
            text,
        )

        try:
            logger.debug("Streaming LLM response for keyword extraction")
            for chunk in self.llm.stream(prompt):
                yield chunk.content
            logger.info("Streaming keyword extraction completed")

        except (ConnectionError, RuntimeError) as e:
            logger.error(f"Error streaming key terms: {str(e)}", exc_info=True)
            yield f"提取关键词时出现错误: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in stream_extract_key_terms: {str(e)}", exc_info=True)
            yield f"提取关键词时出现错误: {str(e)}"

    def stream_extract_actions(self, text: str) -> Iterator[str]:
        """Stream action items extraction for real-time results.

        Args:
            text: The input text to extract action items from

        Yields:
            Streaming chunks of action item extraction results
        """
        logger.info("Streaming action item extraction")

        if not text.strip():
            logger.warning("Empty text provided for streaming action extraction")
            yield "未提供有效文本"
            return

        prompt = self._prepare_prompt(
            "action_extract",
            """请从以下文本中提取需要执行的行动项或任务：

{text}

请列出所有需要执行的任务或行动项，用项目符号列表形式输出。""",
            text,
        )

        try:
            logger.debug("Streaming LLM response for action extraction")
            for chunk in self.llm.stream(prompt):
                yield chunk.content
            logger.info("Streaming action extraction completed")

        except (ConnectionError, RuntimeError) as e:
            logger.error(f"Error streaming action items: {str(e)}", exc_info=True)
            yield f"提取行动项时出现错误: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in stream_extract_actions: {str(e)}", exc_info=True)
            yield f"提取行动项时出现错误: {str(e)}"

    def stream_extract_topics(self, text: str, max_topics: int = 5) -> Iterator[str]:
        """Stream topic extraction for real-time results.

        Args:
            text: The input text to extract topics from
            max_topics: Maximum number of topics to return (default: 5)

        Yields:
            Streaming chunks of topic extraction results
        """
        logger.info(f"Streaming topic extraction (max_topics={max_topics})")

        if not text.strip():
            logger.warning("Empty text provided for streaming topic extraction")
            yield "未提供有效文本"
            return

        prompt = self._prepare_prompt(
            "topic_extract",
            """请从以下文本中提取主要主题：

{text}

请列出文档讨论的主要主题，用项目符号列表形式输出，不需要解释。""",
            text,
        )

        try:
            logger.debug("Streaming LLM response for topic extraction")
            for chunk in self.llm.stream(prompt):
                yield chunk.content
            logger.info("Streaming topic extraction completed")

        except (ConnectionError, RuntimeError) as e:
            logger.error(f"Error streaming topics: {str(e)}", exc_info=True)
            yield f"提取主题时出现错误: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in stream_extract_topics: {str(e)}", exc_info=True)
            yield f"提取主题时出现错误: {str(e)}"