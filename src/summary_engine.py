"""Document summarization engine with custom prompts, streaming, and logging support.

This module provides comprehensive document summarization capabilities with multiple
summary formats and LLM enhancement support. All operations are logged for
monitoring and debugging purposes.
"""

from typing import Iterator, Optional

from langchain_core.language_models import BaseChatModel

from src.prompt_manager import prompt_manager
from src.utils import get_llm
from src.logger import get_logger

# Initialize module logger
logger = get_logger(__name__)


class SummaryEngine:
    """Generates summaries for documents using LLM with enhancement support."""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm if llm is not None else get_llm()
        self._enhancer = None

    def _get_enhancer(self):
        """延迟初始化LLM增强器"""
        if self._enhancer is None:
            from .llm_enhancer import LLMEnhancer
            self._enhancer = LLMEnhancer(self.llm)
        return self._enhancer

    def _get_summary_prompt(self, prompt_name: str, default: str) -> str:
        """Get prompt from file or use default."""
        result = prompt_manager.get_prompt(prompt_name, default)
        return result if result is not None else default

    def generate_summary(self, text: str, length: str = "short", enhance: bool = True) -> str:
        """Generate a summary of the document.
        
        Args:
            text: The input text to summarize
            length: Summary length ('short', 'detailed', 'comprehensive')
            enhance: Whether to enhance summary quality with LLM
        
        Returns:
            Generated summary text
        """

        if not text.strip():
            logger.warning("Empty text provided for summary generation")
            return ""

        prompt_key = (
            f"summary_{length}" if length in ["short", "detailed"] else "summary_short"
        )
        default_prompt = {
            "short": """请对以下文档进行摘要：

{text}

请提供一个简短的摘要，不超过3-5句话。""",
            "detailed": """请对以下文档进行摘要：

{text}

请提供一个详细的摘要，涵盖主要内容和关键点。""",
            "comprehensive": """请对以下文档进行摘要：

{text}

请提供一个全面的摘要，包括所有重要细节。""",
        }

        try:
            prompt = self._get_summary_prompt(
                prompt_key, default_prompt.get(length, default_prompt["short"])
            )
            prompt = prompt.format(text=text[:5000])
            response = self.llm.invoke(prompt)
            summary = response.content
            
            # LLM优化摘要质量
            if enhance and len(summary) > 50:
                enhancer = self._get_enhancer()
                max_length = {"short": 150, "detailed": 300, "comprehensive": 500}.get(length, 150)
                summary = enhancer.enhance_summary(summary, text, max_length)
            
            return summary
        
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}", exc_info=True)
            return ""

    def generate_bullet_summary(self, text: str, enhance: bool = True) -> str:
        """Generate a bullet-point summary.
        
        Args:
            text: The input text to summarize
            enhance: Whether to enhance summary quality with LLM
        
        Returns:
            Generated bullet-point summary
        """

        if not text.strip():
            logger.warning("Empty text provided for bullet summary")
            return ""

        try:
            prompt = self._get_summary_prompt(
                "summary_bullet",
                """请将以下文档内容整理成要点列表：

{text}

请以项目符号列表形式输出主要要点。""",
            )
            prompt = prompt.format(text=text[:5000])
            response = self.llm.invoke(prompt)
            summary = response.content
            
            # LLM优化摘要质量
            if enhance and len(summary) > 50:
                enhancer = self._get_enhancer()
                summary = enhancer.enhance_summary(summary, text, 200)
            
            return summary
        
        except Exception as e:
            logger.error(f"Error generating bullet summary: {str(e)}", exc_info=True)
            return ""

    def generate_executive_summary(self, text: str, enhance: bool = True) -> str:
        """Generate an executive summary.
        
        Args:
            text: The input text to summarize
            enhance: Whether to enhance summary quality with LLM
        
        Returns:
            Generated executive summary
        """

        if not text.strip():
            logger.warning("Empty text provided for executive summary")
            return ""

        try:
            prompt = self._get_summary_prompt(
                "summary_executive",
                """请为以下文档生成一份执行摘要：

{text}

执行摘要应包括：
- 文档的核心目的
- 主要发现或结论
- 关键建议或行动项
- 重要数据或指标

请用简洁专业的语言表达。""",
            )
            prompt = prompt.format(text=text[:5000])
            response = self.llm.invoke(prompt)
            summary = response.content
            
            # LLM优化摘要质量
            if enhance and len(summary) > 50:
                enhancer = self._get_enhancer()
                summary = enhancer.enhance_summary(summary, text, 300)
            
            return summary
        
        except Exception as e:
            logger.error(f"Error generating executive summary: {str(e)}", exc_info=True)
            return ""

    def generate_summary_with_questions(self, text: str, enhance: bool = True) -> str:
        """Generate a summary with key questions and answers.
        
        Args:
            text: The input text to summarize
            enhance: Whether to enhance summary quality with LLM
        
        Returns:
            Generated Q&A summary
        """

        if not text.strip():
            logger.warning("Empty text provided for Q&A summary")
            return ""

        try:
            prompt = self._get_summary_prompt(
                "summary_qa",
                """请阅读以下文档并回答关键问题：

{text}

请回答以下问题：
1. 文档的主要主题是什么？
2. 文档解决了什么问题？
3. 有哪些关键数据或发现？
4. 有什么建议或行动项？
5. 文档的结论是什么？

请以清晰的问答形式输出。""",
            )
            prompt = prompt.format(text=text[:5000])
            response = self.llm.invoke(prompt)
            summary = response.content
            
            # LLM优化摘要质量
            if enhance and len(summary) > 50:
                enhancer = self._get_enhancer()
                summary = enhancer.enhance_summary(summary, text, 350)
            
            return summary
        
        except Exception as e:
            logger.error(f"Error generating Q&A summary: {str(e)}", exc_info=True)
            return ""

    def generate_section_summaries(self, sections: list, enhance: bool = True) -> dict:
        """Generate summaries for each section.
        
        Args:
            sections: List of sections with 'title' and 'content' keys
            enhance: Whether to enhance summary quality with LLM
        
        Returns:
            Dictionary of section summaries
        """
        
        summaries = {}
        for section in sections:
            if "title" in section and "content" in section:
                summaries[section["title"]] = self.generate_summary(section["content"], enhance=enhance)
        
        return summaries

    # Streaming methods
    def stream_summary(self, text: str, length: str = "short") -> Iterator[str]:
        """Stream a summary of the document."""
        prompt_key = (
            f"summary_{length}" if length in ["short", "detailed"] else "summary_short"
        )
        default_prompt = {
            "short": """请对以下文档进行摘要：

{text}

请提供一个简短的摘要，不超过3-5句话。""",
            "detailed": """请对以下文档进行摘要：

{text}

请提供一个详细的摘要，涵盖主要内容和关键点。""",
            "comprehensive": """请对以下文档进行摘要：

{text}

请提供一个全面的摘要，包括所有重要细节。""",
        }

        prompt = self._get_summary_prompt(
            prompt_key, default_prompt.get(length, default_prompt["short"])
        )
        
        # Check if prompt has {text} placeholder, otherwise append text
        if "{text}" in prompt:
            prompt = prompt.format(text=text[:5000])
        else:
            prompt = prompt + "\n\n【待摘要文本】\n" + text[:5000]

        try:
            for chunk in self.llm.stream(prompt):
                yield chunk.content
        except (ConnectionError, RuntimeError) as e:
            yield f"生成摘要时出现错误: {str(e)}"

    def stream_bullet_summary(self, text: str) -> Iterator[str]:
        """Stream a bullet-point summary."""
        prompt = self._get_summary_prompt(
            "summary_bullet",
            """请将以下文档内容整理成要点列表：

{text}

请以项目符号列表形式输出主要要点。""",
        )
        
        # Check if prompt has {text} placeholder, otherwise append text
        if "{text}" in prompt:
            prompt = prompt.format(text=text[:5000])
        else:
            prompt = prompt + "\n\n【待摘要文本】\n" + text[:5000]

        try:
            for chunk in self.llm.stream(prompt):
                yield chunk.content
        except (ConnectionError, RuntimeError) as e:
            yield f"生成要点列表时出现错误: {str(e)}"

    def stream_executive_summary(self, text: str) -> Iterator[str]:
        """Stream an executive summary."""
        prompt = self._get_summary_prompt(
            "summary_executive",
            """请为以下文档生成一份执行摘要：

{text}

执行摘要应包括：
- 文档的主要目的
- 关键发现或结论
- 重要建议或行动项

请用简洁专业的语言表达。""",
        )
        
        # Check if prompt has {text} placeholder, otherwise append text
        if "{text}" in prompt:
            prompt = prompt.format(text=text[:5000])
        else:
            prompt = prompt + "\n\n【待摘要文本】\n" + text[:5000]

        try:
            for chunk in self.llm.stream(prompt):
                yield chunk.content
        except (ConnectionError, RuntimeError) as e:
            yield f"生成执行摘要时出现错误: {str(e)}"

    def stream_summary_with_questions(self, text: str) -> Iterator[str]:
        """Stream a summary with key questions and answers."""
        prompt = self._get_summary_prompt(
            "summary_qa",
            """请阅读以下文档并回答关键问题：

{text}

请回答以下问题：
1. 文档的主要主题是什么？
2. 文档解决了什么问题？
3. 有哪些关键数据或发现？
4. 有什么建议或行动项？
5. 文档的结论是什么？

请以清晰的问答形式输出。""",
        )
        
        # Check if prompt has {text} placeholder, otherwise append text
        if "{text}" in prompt:
            prompt = prompt.format(text=text[:5000])
        else:
            prompt = prompt + "\n\n【待摘要文本】\n" + text[:5000]

        try:
            for chunk in self.llm.stream(prompt):
                yield chunk.content
        except (ConnectionError, RuntimeError) as e:
            yield f"生成问答式摘要时出现错误: {str(e)}"
