"""AI Agent tools — wrap existing engines as LangChain-callable tools.

Each tool accesses document context from st.session_state and delegates to
the appropriate engine module. Tools have detailed descriptions to help the
LLM agent decide when to use each one.

All tools return markdown-formatted strings for nice rendering in the UI.
"""
import streamlit as st
from langchain_core.tools import tool


def _get_doc_text() -> str:
    """Safely get current document text from session state."""
    return st.session_state.get("current_document_text", "")


def _get_compare_text() -> str:
    """Safely get comparison document text from session state."""
    return st.session_state.get("compare_document_text", "")


def _get_engine(name: str):
    """Safely get an engine from session state."""
    return st.session_state.get(name)


# ═══════════════════════════════════════════════════════════════
# Tool 1: Document Q&A
# ═══════════════════════════════════════════════════════════════

@tool
def ask_document(question: str) -> str:
    """基于上传的文档内容回答问题。当用户询问文档中的具体内容、事实、数据、概念或任何需要查证文档的问题时使用。例如："文档中关于预算的部分说了什么"、"作者的核心观点是什么"。

    Args:
        question: 用户想了解的具体问题，越具体越好
    """
    doc_text = _get_doc_text()
    if not doc_text:
        return "⚠️ 请先在左侧侧边栏上传文档并点击「开始处理」。"

    engine = _get_engine("qa_engine")
    if not engine or not engine.rag_chain:
        return "⚠️ 文档索引尚未建立，请先在左侧侧边栏点击「开始处理」上传并索引文档。"

    try:
        result = engine.answer(question)
        answer = result.get("answer", "无法获取回答")
        sources = result.get("sources", [])

        parts = [answer]
        if sources:
            parts.append("\n\n---\n**📖 参考来源:**")
            for s in sources[:3]:
                parts.append(f"- {s}")
        return "\n".join(parts)
    except Exception as e:
        return f"❌ 问答出错: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# Tool 2: Summarization
# ═══════════════════════════════════════════════════════════════

@tool
def summarize_document(style: str = "short") -> str:
    """对上传的文档生成摘要。当用户说"总结"、"概述"、"这篇文章讲了什么"、"给我一个摘要"时使用。

    Args:
        style: 摘要风格，可选值：
            - "short": 简短摘要（3-5句话）
            - "detailed": 详细摘要
            - "bullet": 要点列表
            - "executive": 执行摘要（含核心目的、发现、建议）
            - "qa": 问答式摘要（回答关键问题）
    """
    doc_text = _get_doc_text()
    if not doc_text:
        return "⚠️ 请先在左侧侧边栏上传文档并点击「开始处理」。"

    engine = _get_engine("summary_engine")
    if not engine:
        return "⚠️ 摘要引擎未初始化。"

    try:
        if style == "bullet":
            result = engine.generate_bullet_summary(doc_text, enhance=False)
            return f"**📋 要点摘要:**\n\n{result}"
        elif style == "executive":
            result = engine.generate_executive_summary(doc_text, enhance=False)
            return f"**📊 执行摘要:**\n\n{result}"
        elif style == "qa":
            result = engine.generate_summary_with_questions(doc_text, enhance=False)
            return f"**❓ 问答式摘要:**\n\n{result}"
        elif style == "detailed":
            result = engine.generate_summary(doc_text, length="detailed", enhance=False)
            return f"**📝 详细摘要:**\n\n{result}"
        else:
            result = engine.generate_summary(doc_text, length="short", enhance=False)
            return f"**📝 摘要:**\n\n{result}"
    except Exception as e:
        return f"❌ 摘要生成出错: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# Tool 3: Structure Analysis
# ═══════════════════════════════════════════════════════════════

@tool
def analyze_structure() -> str:
    """分析文档的章节结构、层级关系和组织逻辑。当用户说"分析结构"、"文档大纲"、"有几个章节"、"文档是怎么组织的"、"有目录吗"时使用。
    """
    doc_text = _get_doc_text()
    if not doc_text:
        return "⚠️ 请先在左侧侧边栏上传文档并点击「开始处理」。"

    analyzer = _get_engine("structure_analyzer")
    if not analyzer:
        return "⚠️ 结构分析器未初始化。"

    try:
        result = analyzer.analyze_document(doc_text, use_llm=True)
        if not result.get("has_structure"):
            return "📄 该文档没有明显的层级结构（可能为纯文本或散文格式）。"

        parts = []

        if result.get("doc_type"):
            parts.append(f"**📄 文档类型:** {result['doc_type']}")

        if result.get("doc_purpose"):
            parts.append(f"**🎯 核心目的:** {result['doc_purpose']}")

        parts.append(
            f"**📊 统计:** {result.get('total_sections', 0)} 个章节, "
            f"深度 {result.get('depth', 0)} 层"
        )

        if result.get("overview"):
            parts.append(f"\n**🧠 行文脉络:**\n{result['overview']}")

        # Structure tree
        tree = analyzer.format_tree(result.get("structure_tree", []))
        if tree.strip():
            parts.append(f"\n**📑 目录结构:**\n```\n{tree}\n```")

        # Quality assessment
        quality = result.get("quality", {})
        if quality and quality.get("level"):
            badge = {"high": "🟢 优秀", "medium": "🟡 良好", "low": "🔴 需改进"}
            parts.append(f"\n**🏷️ 质量评级:** {badge.get(quality['level'], quality['level'])}")

        return "\n".join(parts)
    except Exception as e:
        return f"❌ 结构分析出错: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# Tool 4: Info Extraction (keywords / actions / topics)
# ═══════════════════════════════════════════════════════════════

@tool
def extract_info(target: str = "keywords") -> str:
    """从文档中提取结构化信息。当用户说"提取关键词"、"有哪些行动项"、"文档的主题是什么"时使用。

    Args:
        target: 提取目标，可选值：
            - "keywords": 关键词/关键术语
            - "actions": 行动项/待办任务
            - "topics": 主要主题
    """
    doc_text = _get_doc_text()
    if not doc_text:
        return "⚠️ 请先在左侧侧边栏上传文档并点击「开始处理」。"

    extractor = _get_engine("keyword_extractor")
    if not extractor:
        return "⚠️ 提取引擎未初始化。"

    try:
        if target == "actions":
            items = extractor.extract_actions(doc_text, validate=True)
            if not items:
                return "📋 未检测到明确的行动项。"
            lines = [f"- {a}" for a in items]
            return f"**✅ 行动项 ({len(items)} 项):**\n" + "\n".join(lines)
        elif target == "topics":
            items = extractor.extract_topics(doc_text, max_topics=5)
            if not items:
                return "🏷️ 未检测到明确主题。"
            return f"**🏷️ 主要主题:**\n" + "\n".join(f"- {t}" for t in items)
        else:
            items = extractor.extract_key_terms(doc_text, max_terms=10)
            if not items:
                return "🔑 未提取到关键词。"
            return f"**🔑 关键词:** {', '.join(items)}"
    except Exception as e:
        return f"❌ 信息提取出错: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# Tool 5: Translation
# ═══════════════════════════════════════════════════════════════

@tool
def translate_text(text: str = "", target_language: str = "English") -> str:
    """将文本翻译成目标语言。当用户说"翻译成英文"、"翻成日语"、"translate"时使用。如果未指定文本，默认翻译整个文档。

    Args:
        text: 要翻译的文本（留空则翻译整个上传文档）
        target_language: 目标语言名称，如 '中文', 'English', '日本語', '한국어', 'Français', 'Deutsch', 'Español', 'Русский'
    """
    if not text:
        doc_text = _get_doc_text()
        if not doc_text:
            return "⚠️ 请提供要翻译的文本，或先上传文档。"
        text = doc_text[:8000]  # Limit for single translation

    lang_map = {
        "中文": "zh", "English": "en", "日本語": "ja", "한국어": "ko",
        "Français": "fr", "Deutsch": "de", "Español": "es", "Русский": "ru",
    }
    code = lang_map.get(target_language, "en")
    lang_name = target_language

    translator = _get_engine("translation_engine")
    if not translator:
        return "⚠️ 翻译引擎未初始化。"

    try:
        result = translator.translate(text, target_lang=code)
        translation = result.get("translation", result) if isinstance(result, dict) else str(result)
        return f"**🌍 翻译结果 ({lang_name}):**\n\n{translation}"
    except Exception as e:
        return f"❌ 翻译出错: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# Tool 6: Report Generation
# ═══════════════════════════════════════════════════════════════

@tool
def generate_report(format_type: str = "markdown") -> str:
    """生成上传文档的完整综合分析报告。当用户说"生成报告"、"给我一份分析"、"出个综合报告"时使用。

    Args:
        format_type: 报告格式，可选 "markdown" 或 "text"
    """
    doc_text = _get_doc_text()
    if not doc_text:
        return "⚠️ 请先在左侧侧边栏上传文档并点击「开始处理」。"

    generator = _get_engine("report_generator")
    if not generator:
        return "⚠️ 报告引擎未初始化。"

    try:
        if format_type == "text":
            result = generator.generate_full_report(doc_text)
        else:
            result = generator.generate_markdown_report(doc_text)
        return result
    except Exception as e:
        return f"❌ 报告生成出错: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# Tool 7: Document Comparison
# ═══════════════════════════════════════════════════════════════

@tool
def compare_documents() -> str:
    """对比当前主文档和参考文档，找出差异和相似之处。当用户说"对比一下"、"两个文档有什么不同"、"比较这两个文档"时使用。需要在对比Tab上传第二篇文档。
    """
    text1 = _get_doc_text()
    text2 = _get_compare_text()

    if not text1 or not text2:
        return (
            "⚠️ 需要两篇文档才能对比。\n\n"
            "当前仅有 1 篇文档，请切换到 **🔄 文档对比** 功能页面上传第二篇文档，"
            "然后回来重新发起对比请求。"
        )

    comparer = _get_engine("document_comparer")
    if not comparer:
        return "⚠️ 对比引擎未初始化。"

    try:
        similarity = comparer.calculate_similarity(text1, text2)
        diff = comparer.compare_texts(text1, text2)

        parts = [
            f"**📊 文档相似度:** {similarity}%",
            f"**📈 变更统计:** 新增 {diff['stats']['added_lines']} 行, "
            f"删除 {diff['stats']['removed_lines']} 行, "
            f"未变 {diff['stats']['unchanged_lines']} 行",
        ]

        # If similarity is very high or very low, note it
        if similarity > 90:
            parts.append("\nℹ️ 两篇文档高度相似，仅有少量修改。")
        elif similarity < 30:
            parts.append("\nℹ️ 两篇文档内容差异较大。")

        return "\n".join(parts)
    except Exception as e:
        return f"❌ 对比出错: {str(e)}"


# ── Tool Registry ────────────────────────────────────────────

ALL_TOOLS = [
    ask_document,
    summarize_document,
    analyze_structure,
    extract_info,
    translate_text,
    generate_report,
    compare_documents,
]
