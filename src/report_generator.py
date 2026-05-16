"""Document report generator module with logging support.

This module generates comprehensive reports from document analysis with LLM enhancement
and provides detailed logging for monitoring and debugging.
"""
from typing import Dict, Iterator, Optional

from langchain_core.language_models import BaseChatModel

from src.keyword_extractor import KeywordExtractor
from src.structure_analyzer import StructureAnalyzer
from src.summary_engine import SummaryEngine
from src.utils import get_llm
from src.logger import get_logger

# Initialize module logger
logger = get_logger(__name__)


class ReportGenerator:
    """Generates comprehensive reports from document analysis with LLM enhancement."""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        logger.info("Initializing ReportGenerator")
        self.llm = llm if llm is not None else get_llm()
        self.summary_engine = SummaryEngine(llm=self.llm)
        self.keyword_extractor = KeywordExtractor(llm=self.llm)
        self.structure_analyzer = StructureAnalyzer()
        self._enhancer = None
        logger.debug("ReportGenerator initialized successfully")

    def _get_enhancer(self):
        """延迟初始化LLM增强器"""
        if self._enhancer is None:
            from .llm_enhancer import LLMEnhancer
            self._enhancer = LLMEnhancer(self.llm)
        return self._enhancer

    def generate_full_report(self, text: str, enhance: bool = False) -> str:
        """Generate a comprehensive report including all analysis results.
        
        Args:
            text: The document text to analyze
            enhance: Whether to enhance report quality with LLM
        
        Returns:
            Generated full report text
        """
        logger.info(f"Generating full report (enhance={enhance})")

        if not text.strip():
            logger.warning("Empty text provided for report generation")
            return "无法生成报告：未提供有效文本"

        try:
            # Get document structure
            logger.debug("Analyzing document structure")
            structure = self.structure_analyzer.analyze_document(text)

            # Get summary
            logger.debug("Generating document summary")
            summary = self.summary_engine.generate_summary(text)

            # Get key terms
            logger.debug("Extracting key terms")
            key_terms = self.keyword_extractor.extract_key_terms(text)

            # Generate structured report
            logger.debug("Formatting report")
            report = self._format_report(text, structure, summary, key_terms)

            # LLM优化报告内容
            if enhance:
                logger.debug("Enhancing report quality with LLM")
                enhancer = self._get_enhancer()
                report = enhancer.enhance_report(report, text)

            logger.info(f"Full report generated, length: {len(report)}")
            return report
        
        except Exception as e:
            logger.error(f"Error generating full report: {str(e)}", exc_info=True)
            return f"生成报告时出现错误: {str(e)}"

    def generate_markdown_report(self, text: str, enhance: bool = False) -> str:
        """Generate report in markdown format.
        
        Args:
            text: The document text to analyze
            enhance: Whether to enhance report quality with LLM
        
        Returns:
            Generated markdown report
        """
        logger.info(f"Generating markdown report (enhance={enhance})")

        if not text.strip():
            logger.warning("Empty text provided for markdown report")
            return "# 报告生成失败\n\n无法生成报告：未提供有效文本"

        try:
            structure = self.structure_analyzer.analyze_document(text)
            summary = self.summary_engine.generate_summary(text)
            key_terms = self.keyword_extractor.extract_key_terms(text)

            report = self._format_markdown_report(
                text, structure, summary, key_terms
            )
            
            # LLM优化报告内容
            if enhance:
                logger.debug("Enhancing markdown report quality")
                enhancer = self._get_enhancer()
                report = enhancer.enhance_report(report, text)

            logger.info(f"Markdown report generated, length: {len(report)}")
            return report
        
        except Exception as e:
            logger.error(f"Error generating markdown report: {str(e)}", exc_info=True)
            return f"# 报告生成失败\n\n生成报告时出现错误: {str(e)}"

    def _format_report(
        self, text: str, structure: Dict, summary: str, key_terms: str
    ) -> str:
        """Format the report in plain text format."""
        report = []
        report.append("=" * 60)
        report.append("文档分析报告")
        report.append("=" * 60)

        # Document Info
        report.append("\n【文档信息】")
        report.append(f"文档长度: {len(text)} 字符")
        report.append(f"章节数量: {structure.get('total_sections', 0)}")
        report.append(f"文档深度: {structure.get('depth', 0)}")

        # Summary
        report.append("\n【文档摘要】")
        report.append("-" * 40)
        report.append(summary)

        # Key Terms
        report.append("\n【关键词】")
        report.append("-" * 40)
        report.append(", ".join(key_terms) if isinstance(key_terms, list) else key_terms)

        # Structure
        report.append("\n【文档结构】")
        report.append("-" * 40)
        if structure.get("structure_tree"):
            tree = self.structure_analyzer.format_tree(structure["structure_tree"])
            report.append(tree)

        report.append("\n" + "=" * 60)
        report.append("报告生成完成")
        report.append("=" * 60)

        return "\n".join(report)

    def _format_markdown_report(
        self, text: str, structure: Dict, summary: str, key_terms: str
    ) -> str:
        """Format the report in markdown format."""
        report = []
        report.append("# 📊 文档分析报告")
        report.append("")

        # Document Info
        report.append("## 📋 文档信息")
        report.append("| 属性 | 值 |")
        report.append("|------|-----|")
        report.append(f"| 文档长度 | {len(text)} 字符 |")
        report.append(f"| 章节数量 | {structure.get('total_sections', 0)} |")
        report.append(f"| 文档深度 | {structure.get('depth', 0)} |")
        report.append("")

        # Summary
        report.append("## 📝 文档摘要")
        report.append(summary)
        report.append("")

        # Key Terms
        report.append("## 🔑 关键词")
        report.append(", ".join(key_terms) if isinstance(key_terms, list) else key_terms)
        report.append("")

        # Structure
        report.append("## 🏗️ 文档结构")
        if structure.get("structure_tree"):
            tree = self.structure_analyzer.format_tree(structure["structure_tree"])
            report.append("```")
            report.append(tree)
            report.append("```")
        else:
            report.append("无法提取文档结构")

        report.append("")
        report.append("---")
        report.append("*报告由AI智能文档助手生成*")

        return "\n".join(report)

    def export_report(
        self, text: str, format_type: str = "markdown", file_path: Optional[str] = None, enhance: bool = False
    ) -> str:
        """Export report to file.
        
        Args:
            text: The document text to analyze
            format_type: Report format ('markdown' or 'text')
            file_path: Output file path (without extension)
            enhance: Whether to enhance report quality with LLM
        
        Returns:
            Path to the exported file
        """
        logger.info(f"Exporting report (format={format_type}, enhance={enhance})")

        try:
            if format_type == "markdown":
                content = self.generate_markdown_report(text, enhance=enhance)
                ext = ".md"
            else:
                content = self.generate_full_report(text, enhance=enhance)
                ext = ".txt"

            if file_path:
                full_path = file_path + ext
            else:
                import time

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                full_path = f"report_{timestamp}{ext}"

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(f"Report exported to: {full_path}")
            return full_path
        
        except Exception as e:
            logger.error(f"Error exporting report: {str(e)}", exc_info=True)
            return f"导出报告时出现错误: {str(e)}"

    def stream_generate_full_report(self, text: str) -> Iterator[str]:
        """Stream generate a comprehensive report including all analysis results."""
        # Stream document structure
        yield "📊 正在分析文档结构...\n\n"
        structure = self.structure_analyzer.analyze_document(text)
        yield "✓ 文档结构分析完成\n"
        yield f"  - 章节数量: {structure.get('total_sections', 0)}\n"
        yield f"  - 文档深度: {structure.get('depth', 0)}\n\n"

        # Stream summary
        yield "📝 正在生成文档摘要...\n\n"
        summary_stream = self.summary_engine.stream_summary(text)
        summary = ""
        for chunk in summary_stream:
            summary += chunk
            yield chunk
        yield "\n\n✓ 文档摘要生成完成\n\n"

        # Stream key terms
        yield "🔑 正在提取关键词...\n\n"
        key_terms_stream = self.keyword_extractor.stream_extract_key_terms(text)
        key_terms = ""
        for chunk in key_terms_stream:
            key_terms += chunk
            yield chunk
        yield "\n\n✓ 关键词提取完成\n\n"

        # Stream structure
        yield "🏗️ 正在生成文档结构...\n\n"
        if structure.get("structure_tree"):
            tree = self.structure_analyzer.format_tree(structure["structure_tree"])
            yield "```\n"
            yield tree
            yield "\n```\n"
        else:
            yield "无法提取文档结构\n"
        yield "\n✓ 报告生成完成\n"

    def stream_generate_markdown_report(self, text: str) -> Iterator[str]:
        """Stream generate report in markdown format."""
        yield "# 📊 文档分析报告\n\n"

        # Document Info
        yield "## 📋 文档信息\n"
        yield "| 属性 | 值 |\n"
        yield "|------|-----|\n"
        yield f"| 文档长度 | {len(text)} 字符 |\n"
        
        structure = self.structure_analyzer.analyze_document(text)
        yield f"| 章节数量 | {structure.get('total_sections', 0)} |\n"
        yield f"| 文档深度 | {structure.get('depth', 0)} |\n"
        yield "\n"

        # Summary
        yield "## 📝 文档摘要\n"
        summary_stream = self.summary_engine.stream_summary(text)
        for chunk in summary_stream:
            yield chunk
        yield "\n\n"

        # Key Terms
        yield "## 🔑 关键词\n"
        key_terms_stream = self.keyword_extractor.stream_extract_key_terms(text)
        for chunk in key_terms_stream:
            yield chunk
        yield "\n\n"

        # Structure
        yield "## 🏗️ 文档结构\n"
        if structure.get("structure_tree"):
            tree = self.structure_analyzer.format_tree(structure["structure_tree"])
            yield "```\n"
            yield tree
            yield "\n```\n"
        else:
            yield "无法提取文档结构\n"

        yield "\n---\n"
        yield "*报告由AI智能文档助手生成*\n"
