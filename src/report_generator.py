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
        self.llm = llm if llm is not None else get_llm()
        self.summary_engine = SummaryEngine(llm=self.llm)
        self.keyword_extractor = KeywordExtractor(llm=self.llm)
        self.structure_analyzer = StructureAnalyzer()
        self._enhancer = None

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

        if not text.strip():
            logger.warning("Empty text provided for report generation")
            return "无法生成报告：未提供有效文本"

        try:
            # Get document structure
            structure = self.structure_analyzer.analyze_document(text)

            # Get summary
            summary = self.summary_engine.generate_summary(text)

            # Get key terms
            key_terms = self.keyword_extractor.extract_key_terms(text)

            # Generate structured report
            report = self._format_report(text, structure, summary, key_terms)

            # LLM优化报告内容
            if enhance:
                enhancer = self._get_enhancer()
                report = enhancer.enhance_report(report, text)

            return report
        
        except Exception as e:
            logger.error(f"Error generating full report: {str(e)}", exc_info=True)
            return f"生成报告时出现错误: {str(e)}"

    def generate_markdown_report(
        self, text: str, enhance: bool = False, template: str = "standard"
    ) -> str:
        """Generate report in markdown format.
        
        Args:
            text: The document text to analyze
            enhance: Whether to deep-enhance report with LLM
            template: "simple" / "standard" / "detailed"
        """

        if not text.strip():
            logger.warning("Empty text provided for markdown report")
            return "# 报告生成失败\n\n无法生成报告：未提供有效文本"

        try:
            logger.debug("Generating markdown report, template=%s enhance=%s", template, enhance)
            structure = self.structure_analyzer.analyze_document(text)
            summary = self.summary_engine.generate_summary(text)
            key_terms = self.keyword_extractor.extract_key_terms(text)

            report = self._format_markdown_report(
                text, structure, summary, key_terms, template=template
            )

            if enhance:
                enhancer = self._get_enhancer()
                report = enhancer.enhance_report(report, text)

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
        self, text: str, structure: Dict, summary: str, key_terms: str,
        template: str = "standard"
    ) -> str:
        """Format the report in markdown format.

        Templates:
            simple   — Stats + Summary only
            standard — Stats + Summary + Keywords + Structure + Insights
            detailed — All sections with full detail, meant for LLM polish
        """
        import re
        report = []
        report.append("# 📊 文档分析报告")
        report.append("")

        include_stats = template in ("standard", "detailed", "simple")
        include_summary = True
        include_keywords = template in ("standard", "detailed")
        include_structure = template in ("standard", "detailed")
        include_insights = template in ("standard", "detailed")

        # ── Statistics ──
        if include_stats:
            words = re.findall(r'\b\w+\b', text)
            sentences = [s.strip() for s in re.split(r'[。！？.!?]+', text) if s.strip()]
            avg_sentence_len = round(len(words) / max(len(sentences), 1))
            reading_time = max(1, round(len(words) / 200))

            report.append("## 📊 文档统计")
            report.append("| 指标 | 数值 |")
            report.append("|------|------|")
            report.append(f"| 总字符数 | {len(text):,} |")
            report.append(f"| 总词数（英文） | {len(words):,} |")
            report.append(f"| 句子数 | {len(sentences):,} |")
            report.append(f"| 平均句长 | {avg_sentence_len} 词 |")
            report.append(f"| 预估阅读时间 | {reading_time} 分钟 |")
            report.append(f"| 章节数量 | {structure.get('total_sections', 0)} |")
            if template == "detailed":
                report.append(f"| 文档深度 | {structure.get('depth', 0)} |")
            report.append("")

        # ── Summary ──
        if include_summary:
            report.append("## 📝 内容摘要")
            report.append(summary)
            report.append("")

        # ── Key Terms ──
        if include_keywords:
            report.append("## 🔑 关键词")
            terms = key_terms if isinstance(key_terms, list) else [t.strip() for t in key_terms.replace("，","").split("、") if t.strip()]
            if terms:
                report.append("| 关键词 |")
                report.append("|--------|")
                for t in terms:
                    report.append(f"| {t} |")
            report.append("")

        # ── Structure ──
        if include_structure:
            report.append("## 🏗️ 文档结构")
            if structure.get("structure_tree"):
                tree = self.structure_analyzer.format_tree(structure["structure_tree"])
                report.append("```")
                report.append(tree)
                report.append("```")
            else:
                report.append("无法提取文档结构")
            report.append("")

        # ── Insights ──
        if include_insights:
            doc_type = structure.get("doc_type", "")
            doc_purpose = structure.get("doc_purpose", "")
            quality_info = structure.get("quality", {})
            quality_level = quality_info.get("level", "") if isinstance(quality_info, dict) else ""
            overview = structure.get("overview", "")

            has_insight = doc_type or doc_purpose or quality_level or overview
            if has_insight:
                report.append("## 💡 文档洞察")
                if doc_type:
                    report.append(f"- **文档类型**: {doc_type}")
                if doc_purpose:
                    report.append(f"- **写作目的**: {doc_purpose}")
                if quality_level:
                    badges = {"high": "🟢 高质量", "medium": "🟡 中等质量", "low": "🔴 低质量"}
                    report.append(f"- **质量评级**: {badges.get(quality_level, quality_level)}")
                if overview:
                    report.append(f"- **行文脉络**: {overview}")

                if template == "detailed" and isinstance(quality_info, dict):
                    strengths = quality_info.get("strengths", [])
                    weaknesses = quality_info.get("weaknesses", [])
                    suggestions = quality_info.get("suggestions", [])
                    if strengths:
                        report.append(f"\n**✨ 亮点**")
                        for s in strengths:
                            report.append(f"  - {s}")
                    if weaknesses:
                        report.append(f"\n**⚠️ 不足**")
                        for w in weaknesses:
                            report.append(f"  - {w}")
                    if suggestions:
                        report.append(f"\n**💡 改进建议**")
                        for sg in suggestions:
                            report.append(f"  - {sg}")
                report.append("")

        report.append("---")
        report.append("*报告由AI智能文档助手自动生成*")

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

    def stream_generate_markdown_report(
        self, text: str, template: str = "standard"
    ) -> Iterator[str]:
        """Stream generate report in markdown format."""
        import re
        yield "# 📊 文档分析报告\n\n"

        include_stats = template in ("standard", "detailed", "simple")
        include_keywords = template in ("standard", "detailed")
        include_structure = template in ("standard", "detailed")
        include_insights = template in ("standard", "detailed")

        # ── Stats ──
        if include_stats:
            yield "## 📊 文档统计\n"
            words = re.findall(r'\b\w+\b', text)
            sentences = [s.strip() for s in re.split(r'[。！？.!?]+', text) if s.strip()]
            avg_sentence_len = round(len(words) / max(len(sentences), 1))
            reading_time = max(1, round(len(words) / 200))
            yield "| 指标 | 数值 |\n|------|------|\n"
            yield f"| 总字符数 | {len(text):,} |\n"
            yield f"| 总词数 | {len(words):,} |\n"
            yield f"| 句子数 | {len(sentences):,} |\n"
            yield f"| 平均句长 | {avg_sentence_len} 词 |\n"
            yield f"| 阅读时间 | {reading_time} 分钟 |\n"

        structure = self.structure_analyzer.analyze_document(text)
        if include_stats:
            yield f"| 章节数量 | {structure.get('total_sections', 0)} |\n"
            if template == "detailed":
                yield f"| 文档深度 | {structure.get('depth', 0)} |\n"
            yield "\n"

        # ── Summary ──
        yield "## 📝 内容摘要\n"
        summary_stream = self.summary_engine.stream_summary(text)
        for chunk in summary_stream:
            yield chunk
        yield "\n\n"

        # ── Key Terms ──
        if include_keywords:
            yield "## 🔑 关键词\n"
            yield "| 关键词 |\n|--------|\n"
            key_terms_stream = self.keyword_extractor.stream_extract_key_terms(text)
            terms_buffer = ""
            for chunk in key_terms_stream:
                terms_buffer += chunk
            for t in [x.strip() for x in terms_buffer.replace("，","").split("、") if x.strip()]:
                yield f"| {t} |\n"
            yield "\n"

        # ── Structure ──
        if include_structure:
            yield "## 🏗️ 文档结构\n"
            if structure.get("structure_tree"):
                tree = self.structure_analyzer.format_tree(structure["structure_tree"])
                yield "```\n"
                yield tree
                yield "\n```\n"
            else:
                yield "无法提取文档结构\n"

        # ── Insights ──
        if include_insights:
            doc_type = structure.get("doc_type", "")
            doc_purpose = structure.get("doc_purpose", "")
            quality_info = structure.get("quality", {})
            quality_level = quality_info.get("level", "") if isinstance(quality_info, dict) else ""
            overview = structure.get("overview", "")

            has_insight = doc_type or doc_purpose or quality_level or overview
            if has_insight:
                yield "\n## 💡 文档洞察\n"
                if doc_type:
                    yield f"- **文档类型**: {doc_type}\n"
                if doc_purpose:
                    yield f"- **写作目的**: {doc_purpose}\n"
                if quality_level:
                    badges = {"high": "🟢 高质量", "medium": "🟡 中等质量", "low": "🔴 低质量"}
                    yield f"- **质量评级**: {badges.get(quality_level, quality_level)}\n"
                if overview:
                    yield f"- **行文脉络**: {overview}\n"

                if template == "detailed" and isinstance(quality_info, dict):
                    strengths = quality_info.get("strengths", [])
                    weaknesses = quality_info.get("weaknesses", [])
                    suggestions = quality_info.get("suggestions", [])
                    if strengths:
                        yield "\n**✨ 亮点**\n"
                        for s in strengths:
                            yield f"  - {s}\n"
                    if weaknesses:
                        yield "\n**⚠️ 不足**\n"
                        for w in weaknesses:
                            yield f"  - {w}\n"
                    if suggestions:
                        yield "\n**💡 改进建议**\n"
                        for sg in suggestions:
                            yield f"  - {sg}\n"

        yield "\n---\n"
        yield "*报告由AI智能文档助手自动生成*\n"
