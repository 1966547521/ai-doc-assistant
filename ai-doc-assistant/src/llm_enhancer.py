"""LLM增强服务 - 智能检测与优化模块

This module provides lightweight LLM-based enhancement capabilities for various
document processing tasks, including quality analysis, validation, and optimization.
All operations are logged for monitoring and debugging purposes.
"""

from typing import List, Dict, Optional
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
        logger.info("Initializing LLMEnhancer")
        self.llm = llm
        self.cache = {}
        logger.debug("LLMEnhancer initialized successfully")
    
    def analyze_structure_quality(self, headings: List[Dict]) -> Dict:
        """分析章节结构质量（轻量检测）

        Args:
            headings: List of heading dictionaries with 'level' and 'text' keys

        Returns:
            Dictionary containing quality assessment and suggestions
        """
        logger.info("Analyzing structure quality")
        logger.debug(f"Analyzing {len(headings)} headings")
        
        if not headings:
            logger.debug("No headings provided, returning low quality")
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
            logger.debug("Invoking LLM for structure quality analysis")
            response = self.llm.invoke(prompt)
            result = self._parse_json_response(response.content)
            logger.debug(f"Structure quality analysis result: {result}")
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
        logger.info("Validating keyword relevance")
        logger.debug(f"Validating {len(keywords)} keywords")
        
        if not keywords or len(keywords) > 15:
            logger.debug("Too many or no keywords, returning top 10")
            return keywords[:10]
        
        keywords_str = ", ".join(keywords)
        prompt = f"""判断以下关键词与文本的相关性，保留最相关的5-8个：

文本摘要：{text[:500]}

关键词：{keywords_str}

请仅用逗号分隔输出保留的关键词，不需要解释。
<END>"""
        
        try:
            logger.debug("Invoking LLM for keyword validation")
            response = self.llm.invoke(prompt)
            result = [k.strip() for k in response.content.split(",") if k.strip()]
            logger.debug(f"Keyword validation completed, {len(result)} keywords remaining")
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
        logger.info("Validating action items")
        logger.debug(f"Validating {len(actions)} action items")
        
        if not actions or len(actions) > 10:
            logger.debug("Too many or no actions, returning top 5")
            return actions[:5]
        
        actions_str = "\n".join([f"{i+1}. {a}" for i, a in enumerate(actions)])
        prompt = f"""判断以下行动项是否有效且与文本相关，保留有效的行动项：

文本：{text[:500]}

行动项：
{actions_str}

请仅用换行分隔输出有效的行动项，不需要解释。
<END>"""
        
        try:
            logger.debug("Invoking LLM for action validation")
            response = self.llm.invoke(prompt)
            result = [a.strip() for a in response.content.split("\n") if a.strip()]
            logger.debug(f"Action validation completed, {len(result)} actions remaining")
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
        logger.info("Validating topic relevance")
        logger.debug(f"Validating {len(topics)} topics")
        
        if not topics or len(topics) > 10:
            logger.debug("Too many or no topics, returning top 5")
            return topics[:5]
        
        topics_str = ", ".join(topics)
        prompt = f"""判断以下主题是否准确反映文本内容，保留最准确的3-5个：

文本摘要：{text[:500]}

主题：{topics_str}

请仅用逗号分隔输出保留的主题。
<END>"""
        
        try:
            logger.debug("Invoking LLM for topic validation")
            response = self.llm.invoke(prompt)
            result = [t.strip() for t in response.content.split(",") if t.strip()]
            logger.debug(f"Topic validation completed, {len(result)} topics remaining")
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
        logger.info("Enhancing summary quality")
        logger.debug(f"Summary length: {len(summary)}, max_length: {max_length}")
        
        if len(summary) < 50:
            logger.debug("Summary too short, returning original")
            return summary
        
        prompt = f"""优化以下摘要，使其更简洁且信息完整（不超过{max_length}字）：

原文摘要：{summary[:300]}

文本上下文：{text[:300]}

优化后的摘要：
<END>"""
        
        try:
            logger.debug("Invoking LLM for summary enhancement")
            response = self.llm.invoke(prompt)
            enhanced = response.content.strip()[:max_length + 50]
            logger.debug(f"Summary enhancement completed, new length: {len(enhanced)}")
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
        logger.info("Evaluating answer quality")
        
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
            logger.debug("Invoking LLM for answer evaluation")
            response = self.llm.invoke(prompt)
            result = self._parse_json_response(response.content)
            logger.debug(f"Answer evaluation result: {result}")
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
        logger.info("Checking translation quality")
        
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
            logger.debug("Invoking LLM for translation quality check")
            response = self.llm.invoke(prompt)
            result = self._parse_json_response(response.content)
            logger.debug(f"Translation quality check result: {result}")
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
        logger.info("Detecting document type")
        
        prompt = f"""判断以下文本类型（学术论文/技术文档/报告/新闻/小说/其他）：

{text[:300]}

请仅输出类型名称。
<END>"""
        
        try:
            logger.debug("Invoking LLM for document type detection")
            response = self.llm.invoke(prompt)
            result = response.content.strip()
            logger.debug(f"Document type detected: {result}")
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
        logger.info("Generating improvement suggestions")
        
        prompt = f"""分析以下文本，给出{max_suggestions}条改进建议，每条不超过20字：

{text[:500]}

请用数字列表输出。
<END>"""
        
        try:
            logger.debug("Invoking LLM for improvement suggestions")
            response = self.llm.invoke(prompt)
            lines = response.content.strip().split("\n")[:max_suggestions]
            cleaned = []
            for line in lines:
                line = line.strip()
                line = line.lstrip("0123456789.．、").strip()
                if line and len(line) <= 30:
                    cleaned.append(line)
            logger.debug(f"Generated {len(cleaned)} improvement suggestions")
            return cleaned
        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {str(e)}", exc_info=True)
            return []
    
    def enhance_report(self, report: str, text: str) -> str:
        """优化报告内容（中等消耗）

        Args:
            report: Original report to enhance
            text: Context text for reference

        Returns:
            Enhanced report text
        """
        logger.info("Enhancing report quality")
        logger.debug(f"Report length: {len(report)}")
        
        if len(report) < 100:
            logger.debug("Report too short, returning original")
            return report
        
        prompt = f"""优化以下报告，使其结构更清晰、内容更专业：

原始报告：{report[:500]}

文档内容：{text[:300]}

优化后的报告（不超过500字）：
<END>"""
        
        try:
            logger.debug("Invoking LLM for report enhancement")
            response = self.llm.invoke(prompt)
            enhanced = response.content.strip()[:600]
            logger.debug(f"Report enhancement completed, new length: {len(enhanced)}")
            return enhanced
        except Exception as e:
            logger.error(f"Error enhancing report: {str(e)}", exc_info=True)
            return report
    
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
        logger.info("Clearing LLMEnhancer cache")
        self.cache.clear()
        logger.debug("Cache cleared successfully")