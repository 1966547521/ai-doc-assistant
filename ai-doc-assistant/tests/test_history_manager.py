"""Tests for HistoryManager functionality."""
import os
import tempfile
import pytest
from src.history_manager import HistoryManager, DocumentHistoryEntry


class TestHistoryManager:
    """Test cases for HistoryManager."""
    
    @pytest.fixture
    def temp_history_file(self):
        """Create a temporary history file for testing."""
        fd, path = tempfile.mkstemp(suffix=".json", prefix="test_history_")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)
    
    @pytest.fixture
    def history_manager(self, temp_history_file):
        """Create a HistoryManager with temporary file."""
        return HistoryManager(temp_history_file)
    
    def test_add_entry(self, history_manager):
        """Test adding an entry to history."""
        history_manager.add_entry(
            filename="test_doc.pdf",
            file_path="/path/to/test_doc.pdf",
            file_size=1024,
            word_count=500,
            chunk_count=5,
            tags=["test", "pdf"]
        )
        
        entries = history_manager.get_all_entries()
        assert len(entries) == 1
        assert entries[0].filename == "test_doc.pdf"
        assert entries[0].file_size == 1024
        assert entries[0].word_count == 500
        assert entries[0].chunk_count == 5
        assert entries[0].tags == ["test", "pdf"]
    
    def test_add_duplicate_entry(self, history_manager):
        """Test that duplicate entries are replaced."""
        history_manager.add_entry(
            filename="test_doc.pdf",
            file_path="/path/to/test_doc.pdf",
            file_size=1024,
            word_count=500,
            chunk_count=5,
            tags=["test"]
        )
        
        history_manager.add_entry(
            filename="test_doc.pdf",
            file_path="/path/to/test_doc.pdf",
            file_size=2048,
            word_count=1000,
            chunk_count=10,
            tags=["updated"]
        )
        
        entries = history_manager.get_all_entries()
        assert len(entries) == 1
        assert entries[0].file_size == 2048
        assert entries[0].tags == ["updated"]
    
    def test_get_entry_by_id(self, history_manager):
        """Test getting an entry by ID."""
        history_manager.add_entry(
            filename="test_doc.pdf",
            file_path="/path/to/test_doc.pdf",
            file_size=1024,
            word_count=500,
            chunk_count=5
        )
        
        entries = history_manager.get_all_entries()
        entry_id = entries[0].id
        
        found_entry = history_manager.get_entry_by_id(entry_id)
        assert found_entry is not None
        assert found_entry.filename == "test_doc.pdf"
    
    def test_delete_entry(self, history_manager):
        """Test deleting an entry."""
        history_manager.add_entry(
            filename="test_doc.pdf",
            file_path="/path/to/test_doc.pdf",
            file_size=1024,
            word_count=500,
            chunk_count=5
        )
        
        entries = history_manager.get_all_entries()
        entry_id = entries[0].id
        
        result = history_manager.delete_entry(entry_id)
        assert result is True
        
        entries = history_manager.get_all_entries()
        assert len(entries) == 0
    
    def test_search_by_name(self, history_manager):
        """Test searching entries by name."""
        history_manager.add_entry(
            filename="AI_Report.pdf",
            file_path="/path/to/AI_Report.pdf",
            file_size=1024,
            word_count=500,
            chunk_count=5
        )
        history_manager.add_entry(
            filename="Finance_Report.pdf",
            file_path="/path/to/Finance_Report.pdf",
            file_size=2048,
            word_count=1000,
            chunk_count=10
        )
        
        results = history_manager.search_by_name("AI")
        assert len(results) == 1
        assert results[0].filename == "AI_Report.pdf"
        
        results = history_manager.search_by_name("Report")
        assert len(results) == 2
    
    def test_update_entry_tags(self, history_manager):
        """Test updating entry tags."""
        history_manager.add_entry(
            filename="test_doc.pdf",
            file_path="/path/to/test_doc.pdf",
            file_size=1024,
            word_count=500,
            chunk_count=5,
            tags=["old_tag"]
        )
        
        entries = history_manager.get_all_entries()
        entry_id = entries[0].id
        
        result = history_manager.update_entry_tags(entry_id, ["new_tag1", "new_tag2"])
        assert result is True
        
        updated_entry = history_manager.get_entry_by_id(entry_id)
        assert updated_entry.tags == ["new_tag1", "new_tag2"]
    
    def test_clear_history(self, history_manager):
        """Test clearing all history."""
        history_manager.add_entry(
            filename="test_doc1.pdf",
            file_path="/path/to/test_doc1.pdf",
            file_size=1024,
            word_count=500,
            chunk_count=5
        )
        history_manager.add_entry(
            filename="test_doc2.pdf",
            file_path="/path/to/test_doc2.pdf",
            file_size=2048,
            word_count=1000,
            chunk_count=10
        )
        
        assert len(history_manager.get_all_entries()) == 2
        
        history_manager.clear_history()
        
        assert len(history_manager.get_all_entries()) == 0
    
    def test_get_recent_entries(self, history_manager):
        """Test getting recent entries."""
        for i in range(15):
            history_manager.add_entry(
                filename=f"doc{i}.pdf",
                file_path=f"/path/to/doc{i}.pdf",
                file_size=1024,
                word_count=500,
                chunk_count=5
            )
        
        recent = history_manager.get_recent_entries(10)
        assert len(recent) == 10
        assert recent[0].filename == "doc14.pdf"  # Most recent
    
    def test_entry_properties(self):
        """Test DocumentHistoryEntry properties."""
        entry = DocumentHistoryEntry(
            id="test_id",
            filename="test_doc.pdf",
            file_path="/path/to/test_doc.pdf",
            processed_at=1620000000,
            file_size=2048,
            word_count=1000,
            chunk_count=10
        )
        
        assert entry.file_size_human == "2.0 KB"
        assert "2021-05-03" in entry.processed_at_str