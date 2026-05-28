"""Tests for DocumentComparer — diff, similarity, highlight, common sections."""
from unittest.mock import Mock, patch

import pytest
from src.document_comparer import DocumentComparer


@pytest.fixture
def comparer():
    return DocumentComparer(llm=Mock())


# ── calculate_similarity ───────────────────────────────────────────────

class TestCalculateSimilarity:
    def test_identical_texts(self, comparer):
        sim = comparer.calculate_similarity("Hello World", "Hello World")
        assert sim == 100.0

    def test_completely_different(self, comparer):
        sim = comparer.calculate_similarity("abc", "xyz")
        assert sim < 10.0

    def test_partially_similar(self, comparer):
        sim = comparer.calculate_similarity(
            "The quick brown fox", "The quick brown dog"
        )
        assert 50.0 < sim < 100.0

    def test_empty_strings(self, comparer):
        assert comparer.calculate_similarity("", "") == 100.0
        assert comparer.calculate_similarity("abc", "") < 100.0


# ── compare_texts ──────────────────────────────────────────────────────

class TestCompareTexts:
    def test_identical(self, comparer):
        result = comparer.compare_texts("Line1\nLine2", "Line1\nLine2")
        assert result["stats"]["added_lines"] == 0
        assert result["stats"]["removed_lines"] == 0
        assert result["stats"]["unchanged_lines"] == 2
        assert result["similarity"] == 100.0

    def test_added_lines(self, comparer):
        result = comparer.compare_texts("Line1", "Line1\nLine2\nLine3")
        assert result["stats"]["added_lines"] == 2
        assert result["stats"]["unchanged_lines"] >= 1

    def test_removed_lines(self, comparer):
        result = comparer.compare_texts("Line1\nLine2\nLine3", "Line1")
        assert result["stats"]["removed_lines"] == 2
        assert result["stats"]["unchanged_lines"] >= 1

    def test_added_and_removed(self, comparer):
        t1 = "Line A\nLine B\nLine C"
        t2 = "Line A\nLine X\nLine C"
        result = comparer.compare_texts(t1, t2)
        assert result["stats"]["added_lines"] >= 1
        assert result["stats"]["removed_lines"] >= 1

    def test_empty_first(self, comparer):
        result = comparer.compare_texts("", "Line1\nLine2")
        assert result["stats"]["added_lines"] == 2

    def test_empty_second(self, comparer):
        result = comparer.compare_texts("Line1\nLine2", "")
        assert result["stats"]["removed_lines"] == 2

    def test_both_empty(self, comparer):
        result = comparer.compare_texts("", "")
        assert result["stats"]["total_lines"] == 0

    def test_result_shape(self, comparer):
        result = comparer.compare_texts("a\nb", "a\nc")
        assert "added" in result
        assert "removed" in result
        assert "similarity" in result
        assert "stats" in result
        assert isinstance(result["similarity"], float)


# ── generate_diff_summary ──────────────────────────────────────────────

class TestGenerateDiffSummary:
    def test_llm_success(self, comparer):
        comparer.llm.invoke.return_value.content = "这是一份差异总结。"
        result = comparer.generate_diff_summary("abc", "def")
        assert "差异总结" in result

    def test_llm_fallback_on_error(self, comparer):
        comparer.llm.invoke.side_effect = RuntimeError("LLM failed")
        result = comparer.generate_diff_summary("abc", "def")
        assert "相似度" in result
        assert "新增" in result


# ── highlight_differences ──────────────────────────────────────────────

class TestHighlightDifferences:
    def test_identical(self, comparer):
        h1, h2 = comparer.highlight_differences("Line1\nLine2", "Line1\nLine2")
        assert "style" not in h1  # no highlight tags
        assert "style" not in h2

    def test_added_highlighted_in_html2(self, comparer):
        h1, h2 = comparer.highlight_differences("Line1", "Line1\nLine2")
        assert "ccffcc" in h2  # green for added
        assert h1.count("<br>") == h2.count("<br>")

    def test_removed_highlighted_in_html1(self, comparer):
        h1, h2 = comparer.highlight_differences("Line1\nLine2", "Line1")
        assert "ffcccc" in h1  # red for removed

    def test_changed_pair(self, comparer):
        h1, h2 = comparer.highlight_differences("Old text", "New text")
        assert "ffcccc" in h1  # old in red + strikethrough
        assert "ccffcc" in h2  # new in green

    def test_empty_texts(self, comparer):
        h1, h2 = comparer.highlight_differences("", "")
        assert h1 == ""
        assert h2 == ""


# ── find_common_sections ───────────────────────────────────────────────

class TestFindCommonSections:
    def test_finds_common_paragraph(self, comparer):
        common = "这是两篇文档中都存在的段落内容。"
        t1 = f"开头\n\n{common}\n\n结尾"
        t2 = f"其他内容\n\n{common}\n\n收尾"
        result = comparer.find_common_sections(t1, t2)
        assert len(result) >= 1
        assert result[0]["similarity"] > 80

    def test_no_common(self, comparer):
        t1 = "人工智能技术正在快速发展。\n\n深度学习模型在很多领域都取得了突破。"
        t2 = "今天天气很好。\n\n明天可能会下雨。"
        result = comparer.find_common_sections(t1, t2)
        assert len(result) == 0

    def test_short_paragraph_skipped(self, comparer):
        t1 = "a"
        t2 = "a"
        result = comparer.find_common_sections(t1, t2)
        assert len(result) == 0

    def test_returns_sorted_by_similarity(self, comparer):
        t1 = "AAA AAA AAA\n\nBBB BBB BBB\n\nCCC CCC CCC"
        t2 = "CCC CCC CCC\n\nBBB BBB BBB\n\nAAA AAA AAA"
        result = comparer.find_common_sections(t1, t2)
        if result:
            sims = [r["similarity"] for r in result]
            assert sims == sorted(sims, reverse=True)

    def test_result_keys(self, comparer):
        t1 = "Common paragraph text here.\n\nDifferent text."
        t2 = "Common paragraph text here.\n\nOther text."
        result = comparer.find_common_sections(t1, t2)
        if result:
            item = result[0]
            assert "paragraph_a_index" in item
            assert "paragraph_b_index" in item
            assert "similarity" in item
            assert "content_a" in item
            assert "content_b" in item
