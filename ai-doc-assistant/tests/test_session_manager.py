"""Tests for session_manager module."""
import json
import os
import tempfile
import time
import pytest

from src.session_manager import SessionManager, SessionEntry


class TestSessionEntry:
    """Test cases for SessionEntry dataclass."""
    
    def test_session_entry_creation(self):
        """Test creating a session entry."""
        entry = SessionEntry(
            id="test_123",
            name="Test Session",
            created_at=time.time(),
            updated_at=time.time(),
            documents=[{"filename": "test.pdf"}],
            chat_history=[],
            analysis_results={},
            document_text="Test content",
            document_count=1,
            word_count=100
        )
        
        assert entry.id == "test_123"
        assert entry.name == "Test Session"
        assert entry.document_count == 1
        assert entry.word_count == 100
    
    def test_session_entry_post_init(self):
        """Test post_init initialization."""
        entry = SessionEntry(
            id="test_123",
            name="Test Session",
            created_at=time.time(),
            updated_at=time.time(),
            documents=None,
            chat_history=None,
            analysis_results=None,
            document_text="Test",
            document_count=0,
            word_count=0
        )
        
        assert entry.documents == []
        assert entry.chat_history == []
        assert entry.analysis_results == {}
    
    def test_created_at_str(self):
        """Test formatted creation time."""
        timestamp = time.mktime((2024, 1, 1, 12, 0, 0, 0, 0, 0))
        entry = SessionEntry(
            id="test",
            name="Test",
            created_at=timestamp,
            updated_at=timestamp,
            documents=[],
            chat_history=[],
            analysis_results={},
            document_text="Test",
            document_count=0,
            word_count=0
        )
        
        assert "2024-01-01" in entry.created_at_str
    
    def test_word_count_human(self):
        """Test human-readable word count."""
        entry = SessionEntry(
            id="test",
            name="Test",
            created_at=time.time(),
            updated_at=time.time(),
            documents=[],
            chat_history=[],
            analysis_results={},
            document_text="Test",
            document_count=0,
            word_count=500
        )
        
        assert entry.word_count_human == "500 字"
        
        entry.word_count = 1500
        assert "1.5k" in entry.word_count_human
        
        entry.word_count = 15000
        assert "1.5w" in entry.word_count_human


class TestSessionManager:
    """Test cases for SessionManager class."""
    
    @pytest.fixture
    def temp_sessions_file(self):
        """Create a temporary file for sessions testing."""
        fd, path = tempfile.mkstemp(suffix=".json", prefix="test_sessions_")
        os.close(fd)
        yield path
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                pass
    
    @pytest.fixture
    def session_manager(self, temp_sessions_file):
        """Create a SessionManager instance with temporary file."""
        manager = SessionManager(sessions_file=temp_sessions_file)
        yield manager
        if os.path.exists(temp_sessions_file):
            try:
                os.remove(temp_sessions_file)
            except PermissionError:
                pass
    
    def test_init(self, session_manager):
        """Test SessionManager initialization."""
        assert session_manager.sessions_file.endswith(".json")
        assert isinstance(session_manager.sessions, list)
    
    def test_create_session(self, session_manager):
        """Test creating a new session."""
        documents = [{"filename": "test.pdf", "file_size": 1024}]
        
        session = session_manager.create_session(
            name="Test Session",
            documents=documents,
            document_text="Test document content",
            analysis_results={"total_sections": 5}
        )
        
        assert session.name == "Test Session"
        assert session.document_count == 1
        assert len(session.documents) == 1
        assert session.analysis_results["total_sections"] == 5
        assert session.id in [s.id for s in session_manager.sessions]
    
    def test_create_session_long_name(self, session_manager):
        """Test creating session with long name."""
        long_name = "A" * 50
        
        session = session_manager.create_session(
            name=long_name,
            documents=[],
            document_text="Test",
            analysis_results={}
        )
        
        # SessionManager doesn't truncate names, app.py does that
        assert session.name == long_name
        assert len(session.name) == 50
    
    def test_create_session_limit(self, session_manager):
        """Test that sessions are limited to 50."""
        for i in range(60):
            session_manager.create_session(
                name=f"Session {i}",
                documents=[],
                document_text=f"Content {i}",
                analysis_results={}
            )
        
        assert len(session_manager.sessions) == 50
    
    def test_update_session(self, session_manager):
        """Test updating an existing session."""
        session = session_manager.create_session(
            name="Test",
            documents=[],
            document_text="Test",
            analysis_results={}
        )
        
        chat_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        
        result = session_manager.update_session(
            session.id,
            chat_history=chat_history
        )
        
        assert result is True
        updated_session = session_manager.get_session_by_id(session.id)
        assert len(updated_session.chat_history) == 2
        assert updated_session.chat_history[0]["content"] == "Hello"
    
    def test_update_session_nonexistent(self, session_manager):
        """Test updating non-existent session."""
        result = session_manager.update_session(
            "nonexistent_id",
            chat_history=[]
        )
        
        assert result is False
    
    def test_add_message(self, session_manager):
        """Test adding a message to session."""
        session = session_manager.create_session(
            name="Test",
            documents=[],
            document_text="Test",
            analysis_results={}
        )
        
        result = session_manager.add_message(
            session.id,
            "user",
            "Hello world"
        )
        
        assert result is True
        updated_session = session_manager.get_session_by_id(session.id)
        assert len(updated_session.chat_history) == 1
        assert updated_session.chat_history[0]["role"] == "user"
        assert updated_session.chat_history[0]["content"] == "Hello world"
    
    def test_add_message_nonexistent(self, session_manager):
        """Test adding message to non-existent session."""
        result = session_manager.add_message(
            "nonexistent_id",
            "user",
            "Test"
        )
        
        assert result is False
    
    def test_get_session_by_id(self, session_manager):
        """Test getting session by ID."""
        session = session_manager.create_session(
            name="Test",
            documents=[],
            document_text="Test",
            analysis_results={}
        )
        
        found = session_manager.get_session_by_id(session.id)
        
        assert found is not None
        assert found.id == session.id
        assert found.name == "Test"
    
    def test_get_session_by_id_nonexistent(self, session_manager):
        """Test getting non-existent session."""
        found = session_manager.get_session_by_id("nonexistent")
        assert found is None
    
    def test_get_all_sessions(self, session_manager):
        """Test getting all sessions."""
        session_manager.create_session("Session 1", [], "Test1", {})
        session_manager.create_session("Session 2", [], "Test2", {})
        session_manager.create_session("Session 3", [], "Test3", {})
        
        all_sessions = session_manager.get_all_sessions()
        
        assert len(all_sessions) == 3
        assert all_sessions[0].name == "Session 3"  # Most recent first
    
    def test_delete_session(self, session_manager):
        """Test deleting a session."""
        session = session_manager.create_session(
            name="Test",
            documents=[],
            document_text="Test",
            analysis_results={}
        )
        
        result = session_manager.delete_session(session.id)
        
        assert result is True
        assert session_manager.get_session_by_id(session.id) is None
        assert len(session_manager.sessions) == 0
    
    def test_delete_session_nonexistent(self, session_manager):
        """Test deleting non-existent session."""
        result = session_manager.delete_session("nonexistent")
        assert result is False
    
    def test_clear_sessions(self, session_manager):
        """Test clearing all sessions."""
        session_manager.create_session("Session 1", [], "Test1", {})
        session_manager.create_session("Session 2", [], "Test2", {})
        
        session_manager.clear_sessions()
        
        assert len(session_manager.sessions) == 0
    
    def test_search_by_name(self, session_manager):
        """Test searching sessions by name."""
        session_manager.create_session("Python Programming", [], "Test1", {})
        session_manager.create_session("Java Basics", [], "Test2", {})
        session_manager.create_session("Python Advanced", [], "Test3", {})
        
        results = session_manager.search_by_name("python")
        
        assert len(results) == 2
        assert all("python" in s.name.lower() for s in results)
    
    def test_search_by_name_empty(self, session_manager):
        """Test searching with no results."""
        session_manager.create_session("Session 1", [], "Test", {})
        
        results = session_manager.search_by_name("nonexistent")
        
        assert len(results) == 0
    
    def test_get_recent_sessions(self, session_manager):
        """Test getting recent sessions."""
        for i in range(10):
            session_manager.create_session(f"Session {i}", [], f"Test{i}", {})
        
        recent = session_manager.get_recent_sessions(limit=5)
        
        assert len(recent) == 5
        assert recent[0].name == "Session 9"
    
    def test_persistence(self, temp_sessions_file):
        """Test that sessions persist across instances."""
        manager1 = SessionManager(sessions_file=temp_sessions_file)
        manager1.create_session("Test", [], "Content", {})
        
        manager2 = SessionManager(sessions_file=temp_sessions_file)
        
        assert len(manager2.sessions) == 1
        assert manager2.sessions[0].name == "Test"
    
    def test_save_and_load(self, temp_sessions_file):
        """Test saving and loading sessions."""
        manager = SessionManager(sessions_file=temp_sessions_file)
        
        session = manager.create_session(
            name="Test Session",
            documents=[{"filename": "test.pdf"}],
            document_text="Test content",
            analysis_results={"key": "value"}
        )
        
        manager.add_message(session.id, "user", "Hello")
        
        # Verify file was created
        assert os.path.exists(temp_sessions_file)
        
        # Load and verify content
        with open(temp_sessions_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert len(data) == 1
        assert data[0]["name"] == "Test Session"
        assert len(data[0]["chat_history"]) == 1
