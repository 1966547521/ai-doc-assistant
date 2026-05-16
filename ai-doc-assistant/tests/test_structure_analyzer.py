"""Tests for StructureAnalyzer functionality."""
import pytest
from src.structure_analyzer import StructureAnalyzer


class TestStructureAnalyzer:
    """Test cases for StructureAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a StructureAnalyzer instance."""
        return StructureAnalyzer()
    
    @pytest.fixture
    def sample_markdown(self):
        """Create sample markdown content."""
        return """# 标题一

## 标题二

这是标题二的内容。

### 标题三

这是标题三的内容。

## 标题四

这是标题四的内容。"""
    
    def test_extract_headings_markdown(self, analyzer):
        """Test extracting markdown headings."""
        text = "# 主标题\n## 副标题\n### 三级标题\n普通文本"
        headings = analyzer.extract_headings(text)
        
        assert len(headings) == 3
        assert headings[0]["text"] == "主标题"
        assert headings[0]["level"] == "h1"
        assert headings[1]["text"] == "副标题"
        assert headings[1]["level"] == "h2"
        assert headings[2]["text"] == "三级标题"
        assert headings[2]["level"] == "h3"
    
    def test_extract_headings_chinese_brackets(self, analyzer):
        """Test extracting headings with Chinese brackets."""
        text = "【第一章】\n一些内容\n【第二节】\n更多内容"
        headings = analyzer.extract_headings(text)
        
        assert len(headings) == 2
        assert headings[0]["text"] == "第一章"
        assert headings[0]["level"] == "h2"
        assert headings[1]["text"] == "第二节"
        assert headings[1]["level"] == "h2"
    
    def test_extract_headings_numbered(self, analyzer):
        """Test extracting numbered headings."""
        text = "1. 概述\n1.1 背景\n1.2 目的\n2. 方法"
        headings = analyzer.extract_headings(text)
        
        assert len(headings) >= 3
        assert headings[0]["text"] == "概述"
        assert headings[0]["level"] == "h2"
    
    def test_extract_headings_colon(self, analyzer):
        """Test extracting headings with colon separator."""
        text = "第一章：引言\n第二章：方法\n普通文本"
        headings = analyzer.extract_headings(text)
        
        assert len(headings) >= 2
        assert headings[0]["text"] == "第一章"
        assert headings[0]["level"] == "h3"
    
    def test_extract_headings_empty(self, analyzer):
        """Test extracting headings from empty text."""
        headings = analyzer.extract_headings("")
        assert len(headings) == 0
    
    def test_build_document_tree(self, analyzer):
        """Test building document tree from headings."""
        headings = [
            {"text": "标题一", "level": "h1", "line_number": 1},
            {"text": "小标题1", "level": "h2", "line_number": 2},
            {"text": "小标题2", "level": "h2", "line_number": 3},
            {"text": "子标题", "level": "h3", "line_number": 4},
        ]
        
        tree = analyzer.build_document_tree(headings)
        
        assert len(tree) == 1
        assert tree[0]["text"] == "标题一"
        assert len(tree[0]["children"]) == 2
        assert tree[0]["children"][0]["text"] == "小标题1"
        assert tree[0]["children"][1]["text"] == "小标题2"
        assert len(tree[0]["children"][1]["children"]) == 1
        assert tree[0]["children"][1]["children"][0]["text"] == "子标题"
    
    def test_build_document_tree_flat(self, analyzer):
        """Test building tree with no hierarchy."""
        headings = [
            {"text": "项目一", "level": "h2", "line_number": 1},
            {"text": "项目二", "level": "h2", "line_number": 2},
            {"text": "项目三", "level": "h2", "line_number": 3},
        ]
        
        tree = analyzer.build_document_tree(headings)
        
        assert len(tree) == 3
        for node in tree:
            assert len(node["children"]) == 0
    
    def test_build_document_tree_empty(self, analyzer):
        """Test building tree from empty headings list."""
        tree = analyzer.build_document_tree([])
        assert tree == []
    
    def test_extract_sections(self, analyzer):
        """Test extracting document sections."""
        text = """# 第一部分
这是第一部分的内容。

## 细节
这是细节内容。

# 第二部分
这是第二部分的内容。"""
        
        sections, preamble = analyzer.extract_sections(text)
        
        assert len(sections) >= 3
        assert sections[0]["title"] == "第一部分"
        assert "第一部分的内容" in str(sections[0]["content"])
        assert isinstance(preamble, str)
    
    def test_extract_sections_with_preamble(self, analyzer):
        """Test extracting sections with preamble content."""
        text = """这是文档的前言内容。
包含一些介绍性文字。

# 第一章
这是第一章的内容。"""
        
        sections, preamble = analyzer.extract_sections(text)
        
        assert len(sections) == 1
        assert sections[0]["title"] == "第一章"
        assert "文档的前言内容" in preamble
        assert len(preamble) > 0
    
    def test_validate_heading_levels(self, analyzer):
        """Test heading level validation."""
        headings = [
            {"text": "H1", "level": "h1", "line_number": 1},
            {"text": "H3", "level": "h3", "line_number": 2},
        ]
        
        result = analyzer.validate_heading_levels(headings)
        
        assert result["has_issues"] is True
        assert "层级跳跃" in result["summary"]
        assert len(result["details"]) > 0
    
    def test_validate_heading_levels_duplicate(self, analyzer):
        """Test validation of duplicate headings."""
        headings = [
            {"text": "第一章", "level": "h1", "line_number": 1},
            {"text": "第一章", "level": "h1", "line_number": 2},
        ]
        
        result = analyzer.validate_heading_levels(headings)
        
        assert result["has_issues"] is True
        assert "重复" in result["summary"]
        assert len(result["details"]) > 0
    
    def test_validate_heading_levels_valid(self, analyzer):
        """Test validation with valid heading levels."""
        headings = [
            {"text": "H1", "level": "h1", "line_number": 1},
            {"text": "H2", "level": "h2", "line_number": 2},
            {"text": "H3", "level": "h3", "line_number": 3},
        ]
        
        result = analyzer.validate_heading_levels(headings)
        
        assert result["has_issues"] is False
        assert "结构良好" in result["summary"]
        assert len(result["details"]) == 0
    
    def test_calculate_section_stats(self, analyzer):
        """Test section statistics calculation."""
        sections = [
            {"title": "第一章", "level": "h1", "start_line": 1, "content": "短内容"},
            {"title": "第二章", "level": "h1", "start_line": 2, "content": "这是较长的一段内容，包含更多的文字"},
        ]
        
        stats = analyzer.calculate_section_stats(sections)
        
        assert stats["total_sections"] == 2
        assert stats["avg_length"] > 0
        assert stats["max_length"] > stats["min_length"]
        assert stats["longest_section"] == "第二章"
        assert stats["shortest_section"] == "第一章"
        assert stats["total_chars"] > 0
    
    def test_calculate_section_stats_empty(self, analyzer):
        """Test section stats with empty sections."""
        stats = analyzer.calculate_section_stats([])
        
        assert stats["total_sections"] == 0
        assert stats["avg_length"] == 0
        assert stats["max_length"] == 0
        assert stats["min_length"] == 0
        assert stats["longest_section"] is None
        assert stats["shortest_section"] is None
    
    def test_generate_toc_markdown(self, analyzer):
        """Test generating Markdown table of contents."""
        sections = [
            {"title": "第一章", "level": "h1", "start_line": 1, "content": ""},
            {"title": "1.1 小节", "level": "h2", "start_line": 2, "content": ""},
        ]
        
        toc = analyzer.generate_toc(sections, format_type="markdown")
        
        assert "目录" in toc
        assert "第一章" in toc
        assert "1.1 小节" in toc
        assert "#" in toc
    
    def test_generate_toc_text(self, analyzer):
        """Test generating plain text table of contents."""
        sections = [
            {"title": "第一章", "level": "h1", "start_line": 1, "content": ""},
        ]
        
        toc = analyzer.generate_toc(sections, format_type="text")
        
        assert "目录" in toc
        assert "第一章" in toc
        assert "=" in toc
    
    def test_generate_toc_html(self, analyzer):
        """Test generating HTML table of contents."""
        sections = [
            {"title": "第一章", "level": "h1", "start_line": 1, "content": ""},
        ]
        
        toc = analyzer.generate_toc(sections, format_type="html")
        
        assert "<h2>目录</h2>" in toc
        assert "<ul>" in toc
        assert "<li>" in toc
        assert "</ul>" in toc
    
    def test_generate_toc_empty(self, analyzer):
        """Test generating TOC with no sections."""
        toc = analyzer.generate_toc([])
        assert toc == "无章节"
    
    def test_get_level_distribution(self, analyzer):
        """Test getting heading level distribution."""
        headings = [
            {"text": "H1", "level": "h1", "line_number": 1},
            {"text": "H2", "level": "h2", "line_number": 2},
            {"text": "H2b", "level": "h2", "line_number": 3},
            {"text": "H3", "level": "h3", "line_number": 4},
        ]
        
        distribution = analyzer.get_level_distribution(headings)
        
        assert distribution["h1"] == 1
        assert distribution["h2"] == 2
        assert distribution["h3"] == 1
        assert distribution["h4"] == 0
    
    def test_analyze_document(self, analyzer, sample_markdown):
        """Test comprehensive document analysis."""
        result = analyzer.analyze_document(sample_markdown)
        
        assert "headings" in result
        assert "structure_tree" in result
        assert "sections" in result
        assert "preamble" in result
        assert "total_headings" in result
        assert "total_sections" in result
        assert "depth" in result
        assert "validation" in result
        assert "section_stats" in result
        assert "has_preamble" in result
        assert result["total_headings"] >= 3
        assert isinstance(result["structure_tree"], list)
        assert isinstance(result["validation"], dict)
    
    def test_analyze_document_with_preamble(self, analyzer):
        """Test document analysis with preamble."""
        text = """这是前言内容。

# 第一章

这是第一章的内容。"""
        
        result = analyzer.analyze_document(text)
        
        assert result["has_preamble"]
        assert "前言内容" in result["preamble"]
    
    def test_calculate_depth(self, analyzer):
        """Test depth calculation."""
        headings = [
            {"text": "A", "level": "h1", "line_number": 1},
            {"text": "B", "level": "h2", "line_number": 2},
            {"text": "C", "level": "h3", "line_number": 3},
            {"text": "D", "level": "h4", "line_number": 4},
        ]
        
        analyzer.build_document_tree(headings)
        result = analyzer.analyze_document("# A\n## B\n### C\n#### D")
        
        assert result["depth"] >= 3
    
    def test_format_tree(self, analyzer):
        """Test formatting document tree as string."""
        headings = [
            {"text": "标题一", "level": "h1", "line_number": 1},
            {"text": "子标题", "level": "h2", "line_number": 2},
        ]
        
        tree = analyzer.build_document_tree(headings)
        formatted = analyzer.format_tree(tree)
        
        assert "标题一" in formatted
        assert "子标题" in formatted
        assert isinstance(formatted, str)
