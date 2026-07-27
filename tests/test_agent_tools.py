"""Comprehensive tests for all 7 Agent tools with mocked engines."""
from unittest.mock import Mock, patch, MagicMock

from src.agent_tools import (
    ask_document, summarize_document, analyze_structure,
    extract_info, translate_text, generate_report, compare_documents, stream_document_answer,
    stream_summary_document, stream_translate_document, stream_generate_report,
    STREAMING_TOOL_HANDLERS,
    ALL_TOOLS,
)


# ── Mock helpers ──────────────────────────────────────────────────────

def _mock_engine(name, **attrs):
    """Create a mock engine with specified methods."""
    eng = MagicMock()
    for k, v in attrs.items():
        eng.configure_mock(**{k: v})
    return eng


def _patch_helpers(doc_text="", compare_text="", engines=None):
    """Context manager that patches _get_doc_text, _get_compare_text, _get_engine."""
    engines = engines or {}

    def get_engine(name):
        return engines.get(name)

    return patch.multiple(
        "src.agent_tools",
        _get_doc_text=Mock(return_value=doc_text),
        _get_compare_text=Mock(return_value=compare_text),
        _get_engine=Mock(side_effect=get_engine),
    )


# ── 1. ask_document ──────────────────────────────────────────────────

class TestAskDocument:
    def test_no_document(self):
        with _patch_helpers(doc_text=""):
            result = ask_document.invoke({"question": "test"})
        assert "请先" in result

    def test_no_qa_engine(self):
        with _patch_helpers(doc_text="doc content", engines={}):
            result = ask_document.invoke({"question": "test"})
        assert "尚未建立" in result

    def test_no_rag_chain(self):
        qa = _mock_engine("qa_engine", rag_chain=None)
        with _patch_helpers(doc_text="doc content", engines={"qa_engine": qa}):
            result = ask_document.invoke({"question": "test"})
        assert "尚未建立" in result

    def test_success_with_sources(self):
        qa = _mock_engine("qa_engine", rag_chain=Mock())
        qa.answer.return_value = {
            "answer": "答案是42",
            "sources": ["来源1", "来源2", "来源3"],
        }
        with _patch_helpers(doc_text="doc content", engines={"qa_engine": qa}):
            result = ask_document.invoke({"question": "生命的意义"})
        assert "答案是42" in result
        assert "来源1" in result
        assert "来源2" in result
        assert "来源3" in result
        qa.answer.assert_called_once_with("生命的意义")

    def test_success_no_sources(self):
        qa = _mock_engine("qa_engine", rag_chain=Mock())
        qa.answer.return_value = {"answer": "纯文本回答", "sources": []}
        with _patch_helpers(doc_text="doc content", engines={"qa_engine": qa}):
            result = ask_document.invoke({"question": "q"})
        assert "纯文本回答" in result
        assert "参考来源" not in result

    def test_engine_raises_error(self):
        qa = _mock_engine("qa_engine", rag_chain=Mock())
        qa.answer.side_effect = RuntimeError("API挂了")
        with _patch_helpers(doc_text="doc content", engines={"qa_engine": qa}):
            result = ask_document.invoke({"question": "q"})
        assert "API挂了" in result

    def test_streaming_answer_forwards_provider_chunks_and_appends_references(self):
        qa = _mock_engine("qa_engine", rag_chain=Mock())
        qa.stream_answer.return_value = iter(["第一段", "第二段"])
        qa.get_citations.return_value = [{"source_file": "演示.pdf", "page": 2}]
        qa.get_sources.return_value = ["检索片段"]

        with _patch_helpers(doc_text="doc content", engines={"qa_engine": qa}):
            chunks = list(stream_document_answer({"question": "测试问题"}))

        assert chunks[:2] == ["第一段", "第二段"]
        assert "第 2 页" in chunks[-1]
        assert "检索片段" in chunks[-1]
        qa.stream_answer.assert_called_once_with("测试问题")


class TestStreamingAgentTools:
    def test_summary_handler_forwards_engine_deltas(self):
        engine = _mock_engine("summary_engine")
        engine.stream_bullet_summary.return_value = iter(["- 第一项", "\n- 第二项"])

        with _patch_helpers(doc_text="doc content", engines={"summary_engine": engine}):
            chunks = list(stream_summary_document({"style": "bullet"}))

        assert chunks == ["**📋 要点摘要:**\n\n", "- 第一项", "\n- 第二项"]
        engine.stream_bullet_summary.assert_called_once_with("doc content")

    def test_translation_handler_forwards_engine_deltas(self):
        engine = _mock_engine("translation_engine")
        engine.stream_translate.return_value = iter(["Hello", " world"])

        with _patch_helpers(doc_text="中文原文", engines={"translation_engine": engine}):
            chunks = list(stream_translate_document({"text": "", "target_language": "English"}))

        assert chunks == ["**🌍 翻译结果 (English):**\n\n", "Hello", " world"]
        engine.stream_translate.assert_called_once_with("中文原文", target_lang="en")

    def test_report_handler_forwards_markdown_deltas(self):
        engine = _mock_engine("report_generator")
        engine.stream_generate_markdown_report.return_value = iter(["# 报告", "\n内容"])

        with _patch_helpers(doc_text="doc content", engines={"report_generator": engine}):
            chunks = list(stream_generate_report({"format_type": "markdown", "template": "standard"}))

        assert chunks == ["# 报告", "\n内容"]
        engine.stream_generate_markdown_report.assert_called_once_with(
            "doc content", template="standard"
        )

    def test_streaming_handler_registry_covers_all_llm_generation_tools(self):
        assert set(STREAMING_TOOL_HANDLERS) == {
            "ask_document", "summarize_document", "translate_text", "generate_report"
        }


# ── 2. summarize_document ────────────────────────────────────────────

class TestSummarizeDocument:
    def test_no_document(self):
        with _patch_helpers(doc_text=""):
            result = summarize_document.invoke({"style": "short"})
        assert "请先" in result

    def test_no_summary_engine(self):
        with _patch_helpers(doc_text="doc", engines={}):
            result = summarize_document.invoke({"style": "short"})
        assert "未初始化" in result

    def test_style_short(self):
        eng = _mock_engine("summary_engine")
        eng.generate_summary.return_value = "简短摘要内容"
        with _patch_helpers(doc_text="doc", engines={"summary_engine": eng}):
            result = summarize_document.invoke({"style": "short"})
        assert "简短摘要内容" in result
        assert "摘要" in result

    def test_style_detailed(self):
        eng = _mock_engine("summary_engine")
        eng.generate_summary.return_value = "详细摘要内容"
        with _patch_helpers(doc_text="doc", engines={"summary_engine": eng}):
            result = summarize_document.invoke({"style": "detailed"})
        assert "详细摘要内容" in result
        eng.generate_summary.assert_called_once()

    def test_style_bullet(self):
        eng = _mock_engine("summary_engine")
        eng.generate_bullet_summary.return_value = "- 要点1\n- 要点2"
        with _patch_helpers(doc_text="doc", engines={"summary_engine": eng}):
            result = summarize_document.invoke({"style": "bullet"})
        assert "要点1" in result
        eng.generate_bullet_summary.assert_called_once()

    def test_style_executive(self):
        eng = _mock_engine("summary_engine")
        eng.generate_executive_summary.return_value = "执行摘要内容"
        with _patch_helpers(doc_text="doc", engines={"summary_engine": eng}):
            result = summarize_document.invoke({"style": "executive"})
        assert "执行摘要内容" in result
        eng.generate_executive_summary.assert_called_once()

    def test_style_qa(self):
        eng = _mock_engine("summary_engine")
        eng.generate_summary_with_questions.return_value = "Q: 问题\nA: 答案"
        with _patch_helpers(doc_text="doc", engines={"summary_engine": eng}):
            result = summarize_document.invoke({"style": "qa"})
        assert "问答式摘要" in result
        eng.generate_summary_with_questions.assert_called_once()

    def test_style_default_to_short(self):
        eng = _mock_engine("summary_engine")
        eng.generate_summary.return_value = "默认摘要"
        with _patch_helpers(doc_text="doc", engines={"summary_engine": eng}):
            result = summarize_document.invoke({})
        assert "默认摘要" in result

    def test_engine_raises_error(self):
        eng = _mock_engine("summary_engine")
        eng.generate_summary.side_effect = RuntimeError("摘要失败")
        with _patch_helpers(doc_text="doc", engines={"summary_engine": eng}):
            result = summarize_document.invoke({"style": "short"})
        assert "摘要失败" in result


# ── 3. analyze_structure ─────────────────────────────────────────────

class TestAnalyzeStructure:
    def test_no_document(self):
        with _patch_helpers(doc_text=""):
            result = analyze_structure.invoke({})
        assert "请先" in result

    def test_no_analyzer(self):
        with _patch_helpers(doc_text="doc", engines={}):
            result = analyze_structure.invoke({})
        assert "未初始化" in result

    def test_no_structure_found(self):
        analyzer = _mock_engine("structure_analyzer")
        analyzer.analyze_document.return_value = {"has_structure": False}
        with _patch_helpers(doc_text="doc", engines={"structure_analyzer": analyzer}):
            result = analyze_structure.invoke({})
        assert "没有明显的层级结构" in result

    def test_complete_structure(self):
        analyzer = _mock_engine("structure_analyzer")
        analyzer.analyze_document.return_value = {
            "has_structure": True,
            "doc_type": "技术文档",
            "doc_purpose": "介绍系统架构",
            "total_sections": 5,
            "depth": 3,
            "overview": "先讲背景，再讲架构",
            "structure_tree": [{"title": "背景", "children": []}],
            "quality": {"level": "high"},
        }
        analyzer.format_tree.return_value = "1. 背景\n2. 架构\n"
        with _patch_helpers(doc_text="doc", engines={"structure_analyzer": analyzer}):
            result = analyze_structure.invoke({})
        assert "技术文档" in result
        assert "系统架构" in result
        assert "5 个章节" in result
        assert "3 层" in result
        assert "行文脉络" in result
        assert "目录结构" in result
        assert "优秀" in result

    def test_partial_structure(self):
        analyzer = _mock_engine("structure_analyzer")
        analyzer.analyze_document.return_value = {
            "has_structure": True,
            "total_sections": 2,
            "depth": 1,
            "structure_tree": [],
        }
        analyzer.format_tree.return_value = ""
        with _patch_helpers(doc_text="doc", engines={"structure_analyzer": analyzer}):
            result = analyze_structure.invoke({})
        assert "2 个章节" in result
        assert "行文脉络" not in result
        assert "质量评级" not in result

    def test_engine_raises_error(self):
        analyzer = _mock_engine("structure_analyzer")
        analyzer.analyze_document.side_effect = RuntimeError("分析失败")
        with _patch_helpers(doc_text="doc", engines={"structure_analyzer": analyzer}):
            result = analyze_structure.invoke({})
        assert "分析失败" in result


# ── 4. extract_info ──────────────────────────────────────────────────

class TestExtractInfo:
    def test_no_document(self):
        with _patch_helpers(doc_text=""):
            result = extract_info.invoke({"target": "keywords"})
        assert "请先" in result

    def test_no_extractor(self):
        with _patch_helpers(doc_text="doc", engines={}):
            result = extract_info.invoke({"target": "keywords"})
        assert "未初始化" in result

    def test_target_keywords(self):
        eng = _mock_engine("keyword_extractor")
        eng.extract_key_terms.return_value = ["RAG", "向量检索", "ReAct"]
        with _patch_helpers(doc_text="doc", engines={"keyword_extractor": eng}):
            result = extract_info.invoke({"target": "keywords"})
        assert "RAG" in result
        assert "向量检索" in result
        eng.extract_key_terms.assert_called_once()

    def test_target_keywords_empty(self):
        eng = _mock_engine("keyword_extractor")
        eng.extract_key_terms.return_value = []
        with _patch_helpers(doc_text="doc", engines={"keyword_extractor": eng}):
            result = extract_info.invoke({"target": "keywords"})
        assert "未提取到" in result

    def test_target_actions(self):
        eng = _mock_engine("keyword_extractor")
        eng.extract_actions.return_value = ["更新文档", "修复bug"]
        with _patch_helpers(doc_text="doc", engines={"keyword_extractor": eng}):
            result = extract_info.invoke({"target": "actions"})
        assert "更新文档" in result
        assert "2 项" in result
        eng.extract_actions.assert_called_once()

    def test_target_actions_empty(self):
        eng = _mock_engine("keyword_extractor")
        eng.extract_actions.return_value = []
        with _patch_helpers(doc_text="doc", engines={"keyword_extractor": eng}):
            result = extract_info.invoke({"target": "actions"})
        assert "未检测到" in result

    def test_target_topics(self):
        eng = _mock_engine("keyword_extractor")
        eng.extract_topics.return_value = ["AI", "文档处理"]
        with _patch_helpers(doc_text="doc", engines={"keyword_extractor": eng}):
            result = extract_info.invoke({"target": "topics"})
        assert "AI" in result
        eng.extract_topics.assert_called_once()

    def test_target_topics_empty(self):
        eng = _mock_engine("keyword_extractor")
        eng.extract_topics.return_value = []
        with _patch_helpers(doc_text="doc", engines={"keyword_extractor": eng}):
            result = extract_info.invoke({"target": "topics"})
        assert "未检测到" in result

    def test_engine_raises_error(self):
        eng = _mock_engine("keyword_extractor")
        eng.extract_key_terms.side_effect = RuntimeError("提取失败")
        with _patch_helpers(doc_text="doc", engines={"keyword_extractor": eng}):
            result = extract_info.invoke({"target": "keywords"})
        assert "提取失败" in result


# ── 5. translate_text ────────────────────────────────────────────────

class TestTranslateText:
    def test_no_text_no_doc(self):
        with _patch_helpers(doc_text=""):
            result = translate_text.invoke({"text": "", "target_language": "English"})
        assert "请提供" in result

    def test_with_text_needs_no_doc(self):
        eng = _mock_engine("translation_engine")
        eng.translate.return_value = {"translation": "Hello world"}
        with _patch_helpers(doc_text="", engines={"translation_engine": eng}):
            result = translate_text.invoke({"text": "你好世界", "target_language": "English"})
        assert "Hello world" in result
        eng.translate.assert_called_once()

    def test_empty_text_uses_doc(self):
        eng = _mock_engine("translation_engine")
        eng.translate.return_value = {"translation": "Document translated"}
        with _patch_helpers(doc_text="中文文档内容在这里", engines={"translation_engine": eng}):
            result = translate_text.invoke({"text": "", "target_language": "English"})
        assert "Document translated" in result

    def test_no_translation_engine(self):
        with _patch_helpers(doc_text="doc", engines={}):
            result = translate_text.invoke({"text": "你好", "target_language": "English"})
        assert "未初始化" in result

    def test_various_languages(self):
        for lang, lang_display in [("English", "English"), ("日本語", "日本語"),
                                    ("中文", "中文"), ("Français", "Français")]:
            eng = _mock_engine("translation_engine")
            eng.translate.return_value = {"translation": f"translated to {lang}"}
            with _patch_helpers(doc_text="doc", engines={"translation_engine": eng}):
                result = translate_text.invoke({"text": "test", "target_language": lang})
            assert lang_display in result
            assert f"translated to {lang}" in result

    def test_result_is_string_not_dict(self):
        eng = _mock_engine("translation_engine")
        eng.translate.return_value = "直接字符串结果"
        with _patch_helpers(doc_text="doc", engines={"translation_engine": eng}):
            result = translate_text.invoke({"text": "test", "target_language": "English"})
        assert "直接字符串结果" in result

    def test_engine_raises_error(self):
        eng = _mock_engine("translation_engine")
        eng.translate.side_effect = RuntimeError("翻译失败")
        with _patch_helpers(doc_text="doc", engines={"translation_engine": eng}):
            result = translate_text.invoke({"text": "test", "target_language": "English"})
        assert "翻译失败" in result


# ── 6. generate_report ───────────────────────────────────────────────

class TestGenerateReport:
    def test_no_document(self):
        with _patch_helpers(doc_text=""):
            result = generate_report.invoke({"format_type": "markdown", "template": "standard"})
        assert "请先" in result

    def test_no_report_generator(self):
        with _patch_helpers(doc_text="doc", engines={}):
            result = generate_report.invoke({"format_type": "markdown", "template": "standard"})
        assert "未初始化" in result

    def test_markdown_standard(self):
        eng = _mock_engine("report_generator")
        eng.generate_markdown_report.return_value = "# 标准报告\n内容"
        with _patch_helpers(doc_text="doc", engines={"report_generator": eng}):
            result = generate_report.invoke({"format_type": "markdown", "template": "standard"})
        assert "标准报告" in result
        eng.generate_markdown_report.assert_called_once()

    def test_markdown_simple(self):
        eng = _mock_engine("report_generator")
        eng.generate_markdown_report.return_value = "# 简洁报告"
        with _patch_helpers(doc_text="doc", engines={"report_generator": eng}):
            result = generate_report.invoke({"format_type": "markdown", "template": "simple"})
        assert "简洁报告" in result
        _, kwargs = eng.generate_markdown_report.call_args
        assert kwargs.get("enhance") is False

    def test_markdown_detailed(self):
        eng = _mock_engine("report_generator")
        eng.generate_markdown_report.return_value = "# 详尽报告"
        with _patch_helpers(doc_text="doc", engines={"report_generator": eng}):
            result = generate_report.invoke({"format_type": "markdown", "template": "detailed"})
        assert "详尽报告" in result
        _, kwargs = eng.generate_markdown_report.call_args
        assert kwargs.get("enhance") is True

    def test_text_format(self):
        eng = _mock_engine("report_generator")
        eng.generate_full_report.return_value = "纯文本报告内容"
        with _patch_helpers(doc_text="doc", engines={"report_generator": eng}):
            result = generate_report.invoke({"format_type": "text", "template": "standard"})
        assert "纯文本报告内容" in result
        eng.generate_full_report.assert_called_once()

    def test_engine_raises_error(self):
        eng = _mock_engine("report_generator")
        eng.generate_markdown_report.side_effect = RuntimeError("报告生成失败")
        with _patch_helpers(doc_text="doc", engines={"report_generator": eng}):
            result = generate_report.invoke({"format_type": "markdown", "template": "standard"})
        assert "报告生成失败" in result


# ── 7. compare_documents ─────────────────────────────────────────────

class TestCompareDocuments:
    def test_both_docs_missing(self):
        with _patch_helpers(doc_text="", compare_text=""):
            result = compare_documents.invoke({})
        assert "需要两篇" in result

    def test_first_doc_missing(self):
        with _patch_helpers(doc_text="", compare_text="doc2"):
            result = compare_documents.invoke({})
        assert "需要两篇" in result

    def test_second_doc_missing(self):
        with _patch_helpers(doc_text="doc1", compare_text=""):
            result = compare_documents.invoke({})
        assert "需要两篇" in result

    def test_no_comparer(self):
        with _patch_helpers(doc_text="doc1", compare_text="doc2", engines={}):
            result = compare_documents.invoke({})
        assert "未初始化" in result

    def test_normal_similarity(self):
        comparer = _mock_engine("document_comparer")
        comparer.calculate_similarity.return_value = 65
        comparer.compare_texts.return_value = {
            "stats": {"added_lines": 10, "removed_lines": 5, "unchanged_lines": 100},
        }
        with _patch_helpers(doc_text="doc1", compare_text="doc2",
                            engines={"document_comparer": comparer}):
            result = compare_documents.invoke({})
        assert "65%" in result
        assert "10" in result
        assert "5" in result
        assert "100" in result

    def test_high_similarity_flag(self):
        comparer = _mock_engine("document_comparer")
        comparer.calculate_similarity.return_value = 95
        comparer.compare_texts.return_value = {"stats": {"added_lines": 0, "removed_lines": 0, "unchanged_lines": 200}}
        with _patch_helpers(doc_text="doc1", compare_text="doc2",
                            engines={"document_comparer": comparer}):
            result = compare_documents.invoke({})
        assert "高度相似" in result

    def test_low_similarity_flag(self):
        comparer = _mock_engine("document_comparer")
        comparer.calculate_similarity.return_value = 25
        comparer.compare_texts.return_value = {"stats": {"added_lines": 50, "removed_lines": 30, "unchanged_lines": 10}}
        with _patch_helpers(doc_text="doc1", compare_text="doc2",
                            engines={"document_comparer": comparer}):
            result = compare_documents.invoke({})
        assert "差异较大" in result

    def test_engine_raises_error(self):
        comparer = _mock_engine("document_comparer")
        comparer.calculate_similarity.side_effect = RuntimeError("对比失败")
        with _patch_helpers(doc_text="doc1", compare_text="doc2",
                            engines={"document_comparer": comparer}):
            result = compare_documents.invoke({})
        assert "对比失败" in result


# ── Tool Registry ─────────────────────────────────────────────────────

class TestToolRegistry:
    def test_all_tools_registered(self):
        assert len(ALL_TOOLS) == 7
        names = {t.name for t in ALL_TOOLS}
        assert names == {
            "ask_document", "summarize_document", "analyze_structure",
            "extract_info", "translate_text", "generate_report",
            "compare_documents",
        }

    def test_each_tool_has_description(self):
        for tool in ALL_TOOLS:
            assert tool.description, f"{tool.name} has no description"
