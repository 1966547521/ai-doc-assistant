"""Tests for ReportGenerator functionality."""
import os
import pytest
from unittest.mock import Mock, patch
from src.report_generator import ReportGenerator


class TestReportGenerator:
    """Test cases for ReportGenerator."""
    
    @pytest.fixture
    def report_generator(self):
        """Create a ReportGenerator with mocked dependencies."""
        with patch('src.report_generator.get_llm') as mock_get_llm:
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm
            
            with patch('src.report_generator.SummaryEngine') as mock_summary:
                with patch('src.report_generator.KeywordExtractor') as mock_keyword:
                    with patch('src.report_generator.StructureAnalyzer') as mock_structure:
                        mock_summary_instance = Mock()
                        mock_summary_instance.generate_summary.return_value = "Test summary"
                        mock_summary_instance.stream_summary.return_value = iter(["Test", " ", "summary"])
                        mock_summary.return_value = mock_summary_instance
                        
                        mock_keyword_instance = Mock()
                        mock_keyword_instance.extract_key_terms.return_value = ["AI", "Machine Learning", "Python"]
                        mock_keyword_instance.stream_extract_key_terms.return_value = iter(["AI, ", "Machine Learning, ", "Python"])
                        mock_keyword.return_value = mock_keyword_instance
                        
                        mock_structure_instance = Mock()
                        mock_structure_instance.analyze_document.return_value = {
                            "total_sections": 3,
                            "depth": 2,
                            "structure_tree": [
                                {"text": "第一章", "level": "h1", "children": [], "line_number": 1}
                            ],
                        }
                        mock_structure_instance.format_tree.return_value = "- 第一章"
                        mock_structure.return_value = mock_structure_instance
                        
                        generator = ReportGenerator()
                        generator.llm = mock_llm
                        return generator
    
    def test_generate_full_report(self, report_generator):
        """Test generating a full report."""
        result = report_generator.generate_full_report("Test document content.")
        
        assert isinstance(result, str)
        assert "文档分析报告" in result
        assert "文档信息" in result
        assert "文档摘要" in result
        assert "关键词" in result
    
    def test_generate_markdown_report(self, report_generator):
        """Test generating a markdown report."""
        result = report_generator.generate_markdown_report("Test content.")
        
        assert isinstance(result, str)
        assert "# " in result
        assert "## " in result
    
    def test_export_report_markdown(self, report_generator):
        """Test exporting report in markdown format."""
        result_path = report_generator.export_report(
            "Test content.", format_type="markdown",
            file_path="./test_report"
        )
        
        assert result_path == "./test_report.md"
        
        # Cleanup
        if os.path.exists("./test_report.md"):
            os.remove("./test_report.md")
    
    def test_export_report_text(self, report_generator):
        """Test exporting report in text format."""
        result_path = report_generator.export_report(
            "Test content.", format_type="text",
            file_path="./test_report"
        )
        
        assert result_path == "./test_report.txt"
        
        # Cleanup
        if os.path.exists("./test_report.txt"):
            os.remove("./test_report.txt")
    
    def test_export_report_auto_filename(self, report_generator):
        """Test exporting report with auto-generated filename."""
        result_path = report_generator.export_report(
            "Test content.", format_type="markdown"
        )
        
        assert ".md" in result_path
        assert os.path.exists(result_path)
        
        # Cleanup
        if os.path.exists(result_path):
            os.remove(result_path)
    
    def test_format_report_content(self, report_generator):
        """Test that report includes all sections."""
        result = report_generator.generate_full_report("Test content.")
        
        required_sections = [
            "文档分析报告",
            "文档信息",
            "文档摘要",
            "关键词",
            "文档结构",
        ]
        
        for section in required_sections:
            assert section in result, f"Missing section: {section}"
    
    def test_stream_generate_full_report(self, report_generator):
        """Test streaming full report generation."""
        stream = report_generator.stream_generate_full_report("Test content.")
        result = "".join(list(stream))
        
        assert isinstance(result, str)
        assert "正在分析文档结构" in result
        assert "关键词" in result
    
    def test_stream_generate_markdown_report(self, report_generator):
        """Test streaming markdown report generation."""
        stream = report_generator.stream_generate_markdown_report("Test content.")
        result = "".join(list(stream))
        
        assert isinstance(result, str)
        assert "# " in result