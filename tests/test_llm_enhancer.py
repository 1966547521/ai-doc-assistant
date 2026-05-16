"""Tests for LLMEnhancer functionality."""
import pytest
from unittest.mock import Mock

from src.llm_enhancer import LLMEnhancer


@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.invoke.return_value = Mock(content='{"quality": "high", "suggestions": ["add intro", "fix structure"]}')
    return llm


@pytest.fixture
def enhancer(mock_llm):
    return LLMEnhancer(mock_llm)


class TestLLMEnhancer:
    def test_init(self, mock_llm):
        e = LLMEnhancer(mock_llm)
        assert e.llm is mock_llm
        assert e.cache == {}

    def test_analyze_structure_quality(self, enhancer):
        headings = [{"level": "h1", "text": "Introduction"}, {"level": "h2", "text": "Methods"}]
        result = enhancer.analyze_structure_quality(headings)
        assert result["quality"] == "high"
        assert len(result["suggestions"]) == 2

    def test_analyze_structure_quality_empty_headings(self, enhancer):
        result = enhancer.analyze_structure_quality([])
        assert result["quality"] == "low"
        assert result["suggestions"] == []

    def test_analyze_structure_quality_llm_error(self, enhancer, mock_llm):
        mock_llm.invoke.side_effect = RuntimeError("API down")
        headings = [{"level": "h1", "text": "Test"}]
        result = enhancer.analyze_structure_quality(headings)
        assert result["quality"] == "medium"

    def test_validate_keywords(self, enhancer, mock_llm):
        mock_llm.invoke.return_value = Mock(content="AI, ML, NLP, Python")
        result = enhancer.validate_keywords("some text about AI and ML", ["AI", "ML", "NLP", "Python", "Java"])
        assert len(result) <= 8

    def test_validate_keywords_empty(self, enhancer):
        result = enhancer.validate_keywords("text", [])
        assert result == []

    def test_validate_actions(self, enhancer, mock_llm):
        mock_llm.invoke.return_value = Mock(content="\n".join(["Review code", "Write tests"]))
        result = enhancer.validate_actions("we need to review code and write tests",
                                            ["Review code", "Write tests", "Go shopping"])
        assert len(result) <= 5

    def test_validate_actions_empty(self, enhancer):
        result = enhancer.validate_actions("text", [])
        assert result == []

    def test_validate_topics(self, enhancer, mock_llm):
        mock_llm.invoke.return_value = Mock(content="ML, Python")
        result = enhancer.validate_topics("machine learning with python", ["ML", "Python", "Sports"])
        assert len(result) <= 5

    def test_enhance_summary(self, enhancer, mock_llm):
        mock_llm.invoke.return_value = Mock(content="Enhanced short summary text here.")
        result = enhancer.enhance_summary("Original summary text here.", "Context text.", max_length=100)
        assert len(result) > 0

    def test_enhance_summary_too_short(self, enhancer):
        result = enhancer.enhance_summary("short", "context", max_length=50)
        assert result == "short"

    def test_evaluate_answer(self, enhancer, mock_llm):
        mock_llm.invoke.return_value = Mock(
            content='{"relevance":"high","accuracy":"high","completeness":"medium","suggestion":"add details"}'
        )
        result = enhancer.evaluate_answer("What is AI?", "AI is artificial intelligence.", "AI context")
        assert result["relevance"] == "high"
        assert result["accuracy"] == "high"
        assert result["completeness"] == "medium"

    def test_evaluate_answer_default_on_error(self, enhancer, mock_llm):
        mock_llm.invoke.side_effect = RuntimeError("fail")
        result = enhancer.evaluate_answer("q", "a", "c")
        assert result["relevance"] == "medium"

    def test_check_translation_quality(self, enhancer, mock_llm):
        mock_llm.invoke.return_value = Mock(
            content='{"accuracy":"high","fluency":"high","suggestion":""}'
        )
        result = enhancer.check_translation_quality("original", "translated", "English")
        assert result["accuracy"] == "high"

    def test_detect_document_type(self, enhancer, mock_llm):
        mock_llm.invoke.return_value = Mock(content="学术论文")
        result = enhancer.detect_document_type("introduction methods results conclusion")
        assert result == "学术论文"

    def test_detect_document_type_error(self, enhancer, mock_llm):
        mock_llm.invoke.side_effect = RuntimeError("fail")
        result = enhancer.detect_document_type("text")
        assert result == "其他"

    def test_suggest_improvements(self, enhancer, mock_llm):
        mock_llm.invoke.return_value = Mock(content="1. Add examples\n2. Simplify\n3. Add diagrams")
        result = enhancer.suggest_improvements("document text", max_suggestions=3)
        assert len(result) <= 3

    def test_enhance_report(self, enhancer, mock_llm):
        mock_llm.invoke.return_value = Mock(content="Enhanced report " * 20)
        result = enhancer.enhance_report("Original report " * 20, "Context")
        assert len(result) > 0

    def test_enhance_report_too_short(self, enhancer):
        result = enhancer.enhance_report("short", "context")
        assert result == "short"

    def test_parse_json_response(self, enhancer):
        assert enhancer._parse_json_response('{"key": "value"}<END>') == {"key": "value"}
        assert enhancer._parse_json_response('not json') is None

    def test_clear_cache(self, enhancer):
        enhancer.cache["test"] = "value"
        enhancer.clear_cache()
        assert enhancer.cache == {}
