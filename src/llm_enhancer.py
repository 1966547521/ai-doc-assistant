"""LLM增强服务 - 智能检测与优化模块

This module provides lightweight LLM-based enhancement capabilities for various
document processing tasks, including quality analysis, validation, and optimization.
All operations are logged for monitoring and debugging purposes.
"""

from typing import List, Dict, Optional, Iterator
import json

from src.logger import get_logger

# Initialize module logger
logger = get_logger(__name__)


class LLMEnhancer:
    """智能LLM增强服务，为各功能模块提供轻量检测与优化能力

    Features:
        - Structure quality analysis
        - Keyword relevance validation
        - Action item validation
        - Topic validation
        - Summary enhancement
        - Answer quality evaluation
        - Translation quality checking
        - Document type detection
        - Report enhancement
        - Built-in caching for performance optimization
    """
    
    def __init__(self, llm):
        """Initialize the LLMEnhancer.

        Args:
            llm: LLM instance to use for enhancement tasks
        """
        self.llm = llm
        self.cache = {}
    
    def analyze_structure_quality(self, headings: List[Dict]) -> Dict:
        """分析章节结构质量（轻量检测）

        Args:
            headings: List of heading dictionaries with 'level' and 'text' keys

        Returns:
            Dictionary containing quality assessment and suggestions
        """
        
        if not headings:
            return {"quality": "low", "suggestions": []}
        
        headings_text = "\n".join([f"{h['level']}: {h['text']}" for h in headings[:10]])
        prompt = f"""分析以下文档章节结构质量：

{headings_text}

请以JSON格式返回：
{{
    "quality": "high/medium/low",
    "suggestions": ["建议1", "建议2"]
}}

<END>"""
        
        try:
            response = self.llm.invoke(prompt)
            result = self._parse_json_response(response.content)
            return result if result else {"quality": "medium", "suggestions": []}
        except Exception as e:
            logger.error(f"Error analyzing structure quality: {str(e)}", exc_info=True)
            return {"quality": "medium", "suggestions": []}
    
    def validate_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """验证关键词相关性（轻量检测）

        Args:
            text: Original text context
            keywords: List of keywords to validate

        Returns:
            Filtered list of relevant keywords
        """
        
        if not keywords or len(keywords) > 15:
            return keywords[:10]
        
        keywords_str = ", ".join(keywords)
        prompt = f"""判断以下关键词与文本的相关性，保留最相关的5-8个：

文本摘要：{text[:500]}

关键词：{keywords_str}

请仅用逗号分隔输出保留的关键词，不需要解释。
<END>"""
        
        try:
            response = self.llm.invoke(prompt)
            result = [k.strip() for k in response.content.split(",") if k.strip()]
            return result[:8]
        except Exception as e:
            logger.error(f"Error validating keywords: {str(e)}", exc_info=True)
            return keywords[:8]
    
    def validate_actions(self, text: str, actions: List[str]) -> List[str]:
        """验证行动项有效性（轻量检测）

        Args:
            text: Original text context
            actions: List of action items to validate

        Returns:
            Filtered list of valid action items
        """
        
        if not actions or len(actions) > 10:
            return actions[:5]
        
        actions_str = "\n".join([f"{i+1}. {a}" for i, a in enumerate(actions)])
        prompt = f"""判断以下行动项是否有效且与文本相关，保留有效的行动项：

文本：{text[:500]}

行动项：
{actions_str}

请仅用换行分隔输出有效的行动项，不需要解释。
<END>"""
        
        try:
            response = self.llm.invoke(prompt)
            result = [a.strip() for a in response.content.split("\n") if a.strip()]
            return result[:5]
        except Exception as e:
            logger.error(f"Error validating actions: {str(e)}", exc_info=True)
            return actions[:5]
    
    def validate_topics(self, text: str, topics: List[str]) -> List[str]:
        """验证主题相关性（轻量检测）

        Args:
            text: Original text context
            topics: List of topics to validate

        Returns:
            Filtered list of relevant topics
        """
        
        if not topics or len(topics) > 10:
            return topics[:5]
        
        topics_str = ", ".join(topics)
        prompt = f"""判断以下主题是否准确反映文本内容，保留最准确的3-5个：

文本摘要：{text[:500]}

主题：{topics_str}

请仅用逗号分隔输出保留的主题。
<END>"""
        
        try:
            response = self.llm.invoke(prompt)
            result = [t.strip() for t in response.content.split(",") if t.strip()]
            return result[:5]
        except Exception as e:
            logger.error(f"Error validating topics: {str(e)}", exc_info=True)
            return topics[:5]
    
    def enhance_summary(self, summary: str, text: str, max_length: int = 150) -> str:
        """优化摘要质量（中等消耗）

        Args:
            summary: Original summary to enhance
            text: Context text for reference
            max_length: Maximum length of the enhanced summary

        Returns:
            Enhanced summary text
        """
        
        if len(summary) < 50:
            return summary
        
        prompt = f"""优化以下摘要，使其更简洁且信息完整（不超过{max_length}字）：

原文摘要：{summary[:300]}

文本上下文：{text[:300]}

优化后的摘要：
<END>"""
        
        try:
            response = self.llm.invoke(prompt)
            enhanced = response.content.strip()[:max_length + 50]
            return enhanced
        except Exception as e:
            logger.error(f"Error enhancing summary: {str(e)}", exc_info=True)
            return summary
    
    def evaluate_answer(self, question: str, answer: str, context: str) -> Dict:
        """评估问答质量（中等消耗）

        Args:
            question: Original question
            answer: Answer to evaluate
            context: Context used for generating the answer

        Returns:
            Dictionary with relevance, accuracy, completeness, and suggestion
        """
        
        prompt = f"""评估以下回答的质量：

问题：{question[:100]}
回答：{answer[:300]}
上下文：{context[:200]}

请以JSON格式返回评估结果：
{{
    "relevance": "high/medium/low",
    "accuracy": "high/medium/low",
    "completeness": "high/medium/low",
    "suggestion": "改进建议或留空"
}}

<END>"""
        
        try:
            response = self.llm.invoke(prompt)
            result = self._parse_json_response(response.content)
            return result if result else {
                "relevance": "medium", 
                "accuracy": "medium", 
                "completeness": "medium", 
                "suggestion": ""
            }
        except Exception as e:
            logger.error(f"Error evaluating answer: {str(e)}", exc_info=True)
            return {
                "relevance": "medium", 
                "accuracy": "medium", 
                "completeness": "medium", 
                "suggestion": ""
            }
    
    def check_translation_quality(self, original: str, translation: str, target_lang: str) -> Dict:
        """检查翻译质量（轻量检测）

        Args:
            original: Original text
            translation: Translated text
            target_lang: Target language name

        Returns:
            Dictionary with accuracy, fluency, and suggestion
        """
        
        prompt = f"""检查以下翻译的准确性和完整性：

原文：{original[:200]}
翻译（{target_lang}）：{translation[:200]}

请以JSON格式返回：
{{
    "accuracy": "high/medium/low",
    "fluency": "high/medium/low",
    "suggestion": "改进建议或留空"
}}

<END>"""
        
        try:
            response = self.llm.invoke(prompt)
            result = self._parse_json_response(response.content)
            return result if result else {
                "accuracy": "medium", 
                "fluency": "medium", 
                "suggestion": ""
            }
        except Exception as e:
            logger.error(f"Error checking translation quality: {str(e)}", exc_info=True)
            return {
                "accuracy": "medium", 
                "fluency": "medium", 
                "suggestion": ""
            }
    
    def detect_document_type(self, text: str) -> str:
        """检测文档类型（轻量）

        Args:
            text: Text to analyze

        Returns:
            Document type classification
        """
        
        prompt = f"""判断以下文本类型（学术论文/技术文档/报告/新闻/小说/其他）：

{text[:300]}

请仅输出类型名称。
<END>"""
        
        try:
            response = self.llm.invoke(prompt)
            result = response.content.strip()
            return result
        except Exception as e:
            logger.error(f"Error detecting document type: {str(e)}", exc_info=True)
            return "其他"
    
    def suggest_improvements(self, text: str, max_suggestions: int = 3) -> List[str]:
        """建议文档改进方向（轻量）

        Args:
            text: Text to analyze
            max_suggestions: Maximum number of suggestions to return

        Returns:
            List of improvement suggestions
        """
        
        prompt = f"""分析以下文本，给出{max_suggestions}条改进建议，每条不超过20字：

{text[:500]}

请用数字列表输出。
<END>"""
        
        try:
            response = self.llm.invoke(prompt)
            lines = response.content.strip().split("\n")[:max_suggestions]
            cleaned = []
            for line in lines:
                line = line.strip()
                line = line.lstrip("0123456789.．、").strip()
                if line and len(line) <= 30:
                    cleaned.append(line)
            return cleaned
        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {str(e)}", exc_info=True)
            return []
    
    def enhance_report(self, report: str, text: str) -> str:
        """深层润色报告——LLM 阅读全文后生成高质量综合分析报告。

        Args:
            report: 工具生成的 markdown 初稿（含统计数据/摘要/关键词/结构）
            text: 原始文档全文（提供完整上下文）

        Returns:
            经过 LLM 深层分析与润色的 markdown 报告
        """

        if len(report) < 100:
            return report

        text_snippet = text[:4000] if len(text) > 4000 else text

        prompt = f"""你是一位资深文档分析师。请基于下面的统计数据和文档内容，撰写一份专业、有深度的综合分析报告（Markdown格式）。

要求：
1. **保持报告结构**：统计 → 摘要 → 关键词 → 结构分析 → 深度洞察
2. **摘要要提炼核心论点**：不要照搬原文，要用自己的语言总结文档的核心主张和关键论据
3. **关键词要加注解释**：每个关键词后面用括号注明它在文档中的作用
4. **深度洞察至少3条**：包括文档的优势亮点、潜在盲区、改进建议
5. **语言专业但可读**，每条洞察 1-2 句话，不要太长

=== 初稿报告 ===
{report}

=== 文档内容（完整上下文） ===
{text_snippet}

请直接输出 Markdown 格式的报告，不要有任何前言或后语。
"""

        try:
            response = self.llm.invoke(prompt)
            enhanced = response.content.strip()
            # 去掉可能的 markdown 代码块包裹
            if enhanced.startswith("```"):
                lines = enhanced.split("\n")
                enhanced = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            return enhanced
        except Exception as e:
            logger.error(f"Error enhancing report: {str(e)}", exc_info=True)
            return report

    def stream_enhance_report(self, report: str, text: str) -> Iterator[str]:
        """流式深层润色报告——逐 token 返回 LLM 润色结果。

        Args:
            report: 工具生成的 markdown 初稿
            text: 原始文档全文
        Yields:
            润色后的报告文本片段
        """

        if len(report) < 100:
            yield report
            return

        text_snippet = text[:4000] if len(text) > 4000 else text

        prompt = f"""你是一位资深文档分析师。请基于下面的统计数据和文档内容，撰写一份专业、有深度的综合分析报告（Markdown格式）。

要求：
1. **保持报告结构**：统计 → 摘要 → 关键词 → 结构分析 → 深度洞察
2. **摘要要提炼核心论点**：不要照搬原文，要用自己的语言总结文档的核心主张和关键论据
3. **关键词要加注解释**：每个关键词后面用括号注明它在文档中的作用
4. **深度洞察至少3条**：包括文档的优势亮点、潜在盲区、改进建议
5. **语言专业但可读**，每条洞察 1-2 句话，不要太长

=== 初稿报告 ===
{report}

=== 文档内容（完整上下文） ===
{text_snippet}

请直接输出 Markdown 格式的报告，不要有任何前言或后语。
"""

        try:
            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Error streaming report enhancement: {str(e)}", exc_info=True)
            yield report
    
    def _parse_json_response(self, content: str) -> Optional[Dict]:
        """安全解析JSON响应

        Args:
            content: Response content to parse

        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        try:
            if "<END>" in content:
                content = content.split("<END>")[0]
            # 移除可能的markdown代码块标记
            content = content.strip().strip("```json").strip("```").strip()
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {str(e)}")
            return None
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()