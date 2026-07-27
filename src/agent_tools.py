"""AI Agent tools — wrap existing engines as LangChain-callable tools.

Each tool accesses document context from st.session_state and delegates to
the appropriate engine module. Tools have detailed descriptions to help the
LLM agent decide when to use each one.

All tools return markdown-formatted strings for nice rendering in the UI.
"""
import streamlit as st
from langchain_core.tools import tool


def _get_doc_text() -> str:
    """Safely get current document text, with multiple fallbacks."""
    import streamlit as st
    from src.logger import get_logger
    _log = get_logger("agent_tools")

    text = st.session_state.get("current_document_text", "")
    _log.info("_get_doc_text: st.session_state.current_document_text len=%d, sid=%s",
              len(text) if text else 0,
              st.session_state.get("current_session_id"))
    if text:
        return text

    sid = st.session_state.get("current_session_id")
    if sid:
        mgr = st.session_state.get("session_manager")
        if mgr:
            s = mgr.get_session_by_id(sid)
            _log.info("_get_doc_text: session_manager.get_session_by_id(%s) -> %s, doc_len=%d",
                      sid[:20] if sid else None,
                      type(s).__name__ if s else None,
                      len(s.document_text) if (s and s.document_text) else 0)
            if s and s.document_text:
                st.session_state.current_document_text = s.document_text
                st.session_state.documents_uploaded = True
                return s.document_text
        else:
            _log.warning("_get_doc_text: session_manager NOT in session_state")
    else:
        _log.warning("_get_doc_text: current_session_id is None")

    if "current_session_id" not in st.session_state:
        _log.warning("_get_doc_text: no current_session_id key in session_state")
        if hasattr(st, "session_state"):
            _log.info("session_state keys: %s", list(st.session_state.keys())[:20])

    try:
        vs = st.session_state.get("vector_store")
        if vs:
            try:
                store = getattr(vs, "vector_store", None)
                if store is not None:
                    data = store.get()
                    docs = data.get("documents", []) if isinstance(data, dict) else []
                    if docs:
                        joined = "\n\n".join(docs)
                        _log.info("_get_doc_text: recovered %d chars from vector_store", len(joined))
                        st.session_state.current_document_text = joined
                        st.session_state.documents_uploaded = True
                        return joined
            except Exception as e:
                _log.warning("_get_doc_text: vector_store recovery failed: %s", e)
    except Exception:
        pass

    qa = st.session_state.get("qa_engine")
    if qa and getattr(qa, "context_snapshot", None):
        snap = qa.context_snapshot
        _log.info("_get_doc_text: recovered %d chars from qa_engine.context_snapshot", len(snap))
        st.session_state.current_document_text = snap
        st.session_state.documents_uploaded = True
        return snap

    return ""


def _get_compare_text() -> str:
    """Safely get comparison document text from session state."""
    return st.session_state.get("compare_document_text", "")


def _get_engine(name: str):
    """Safely get an engine from session state."""
    return st.session_state.get(name)


def _format_qa_references(citations, sources) -> str:
    """Format the retrieval evidence shared by sync and streaming Q&A."""
    parts = []
    if citations:
        parts.append("\n\n---\n**📍 引用定位:**")
        for citation in citations[:3]:
            location = citation.get("source_file", "文档")
            if "page" in citation:
                location += f"，第 {citation['page']} 页"
            elif "paragraph" in citation:
                location += f"，第 {citation['paragraph']} 段"
            elif "table" in citation and "row" in citation:
                location += f"，表 {citation['table']} 第 {citation['row']} 行"
            parts.append(f"- {location}")
    if sources:
        parts.append("\n\n---\n**📖 参考来源:**")
        parts.extend(f"- {source}" for source in sources[:3])
    return "\n".join(parts)


def stream_document_answer(arguments):
    """Yield real provider chunks for the RAG tool, followed by citations."""
    question = str(arguments.get("question", "")).strip()
    doc_text = _get_doc_text()
    if not doc_text:
        yield "⚠️ 请先在左侧侧边栏上传文档并点击「开始处理」。"
        return

    engine = _get_engine("qa_engine")
    if not engine or not engine.rag_chain:
        yield "⚠️ 文档索引尚未建立，请先在左侧侧边栏点击「开始处理」上传并索引文档。"
        return

    try:
        yield from engine.stream_answer(question)
        citations = engine.get_citations(question)
        sources = engine.get_sources(question)
        references = _format_qa_references(citations, sources)
        if references:
            yield references
    except Exception as exc:
        yield f"❌ 问答出错: {exc}"


def stream_summary_document(arguments):
    """Yield real LLM deltas for document summaries without a rewrite pass."""
    doc_text = _get_doc_text()
    if not doc_text:
        yield "⚠️ 请先在左侧侧边栏上传文档并点击「开始处理」。"
        return

    engine = _get_engine("summary_engine")
    if not engine:
        yield "⚠️ 摘要引擎未初始化。"
        return

    style = arguments.get("style", "short")
    streamers = {
        "bullet": ("**📋 要点摘要:**\n\n", engine.stream_bullet_summary),
        "executive": ("**📊 执行摘要:**\n\n", engine.stream_executive_summary),
        "qa": ("**❓ 问答式摘要:**\n\n", engine.stream_summary_with_questions),
        "detailed": ("**📝 详细摘要:**\n\n", lambda text: engine.stream_summary(text, length="detailed")),
        "short": ("**📝 摘要:**\n\n", lambda text: engine.stream_summary(text, length="short")),
    }
    prefix, streamer = streamers.get(style, streamers["short"])
    yield prefix
    try:
        yield from streamer(doc_text)
    except Exception as exc:
        yield f"❌ 摘要生成出错: {exc}"


def stream_translate_document(arguments):
    """Yield real LLM deltas for a translation request."""
    text = str(arguments.get("text", ""))
    if not text:
        text = _get_doc_text()
    if not text:
        yield "⚠️ 请提供要翻译的文本，或先上传文档。"
        return

    target_language = str(arguments.get("target_language", "English"))
    lang_map = {
        "中文": "zh", "English": "en", "日本語": "ja", "한국어": "ko",
        "Français": "fr", "Deutsch": "de", "Español": "es", "Русский": "ru",
    }
    translator = _get_engine("translation_engine")
    if not translator:
        yield "⚠️ 翻译引擎未初始化。"
        return

    yield f"**🌍 翻译结果 ({target_language}):**\n\n"
    try:
        yield from translator.stream_translate(text, target_lang=lang_map.get(target_language, "en"))
    except Exception as exc:
        yield f"❌ 翻译出错: {exc}"


def stream_generate_report(arguments):
    """Stream report prose while preserving explicit progress for rule-based work."""
    doc_text = _get_doc_text()
    if not doc_text:
        yield "⚠️ 请先在左侧侧边栏上传文档并点击「开始处理」。"
        return

    generator = _get_engine("report_generator")
    if not generator:
        yield "⚠️ 报告引擎未初始化。"
        return

    format_type = arguments.get("format_type", "markdown")
    template = arguments.get("template", "standard")
    try:
        if format_type == "text":
            yield from generator.stream_generate_full_report(doc_text)
        else:
            yield from generator.stream_generate_markdown_report(doc_text, template=template)
    except Exception as exc:
        yield f"❌ 报告生成出错: {exc}"


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
        citations = result.get("citations", [])

        return answer + _format_qa_references(citations, sources)
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
            return "**🏷️ 主要主题:**\n" + "\n".join(f"- {t}" for t in items)
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
def generate_report(format_type: str = "markdown", template: str = "standard") -> str:
    """生成上传文档的完整综合分析报告。当用户说"生成报告"、"给我一份分析"、"出个综合报告"时使用。

    Args:
        format_type: 报告格式，可选 "markdown" 或 "text"
        template: 报告模板，可选 "simple"(简洁)、"standard"(标准)、"detailed"(详尽+AI润色)
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
            enhance = template == "detailed"
            result = generator.generate_markdown_report(
                doc_text, enhance=enhance, template=template
            )
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

STREAMING_TOOL_HANDLERS = {
    "ask_document": stream_document_answer,
    "summarize_document": stream_summary_document,
    "translate_text": stream_translate_document,
    "generate_report": stream_generate_report,
}
