"""Module for managing conversation sessions."""
import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

from src.runtime_paths import SESSIONS_FILE, ensure_runtime_directories


@dataclass
class SessionEntry:
    """Represents a conversation session."""
    id: str
    name: str
    created_at: float
    updated_at: float
    documents: List[Dict[str, Any]]
    chat_history: List[Dict[str, str]]
    analysis_results: Dict[str, Any]
    document_text: str
    document_count: int
    word_count: int
    generated_content: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.documents:
            self.documents = []
        if not self.chat_history:
            self.chat_history = []
        if not self.analysis_results:
            self.analysis_results = {}
        if self.generated_content is None:
            self.generated_content = {}
    
    @property
    def created_at_str(self) -> str:
        """Return formatted creation time."""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created_at))
    
    @property
    def updated_at_str(self) -> str:
        """Return formatted update time."""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.updated_at))
    
    @property
    def word_count_human(self) -> str:
        """Return human-readable word count."""
        if self.word_count < 1000:
            return f"{self.word_count} 字"
        elif self.word_count < 10000:
            return f"{self.word_count / 1000:.1f}k 字"
        else:
            return f"{self.word_count / 10000:.1f}w 字"


class SessionManager:
    """Manages conversation sessions."""
    
    def __init__(self, sessions_file: str = str(SESSIONS_FILE)):
        ensure_runtime_directories()
        self.sessions_file = sessions_file
        self.sessions: List[SessionEntry] = []
        self._index: Dict[str, SessionEntry] = {}
        self._load_sessions()
    
    def _load_sessions(self) -> None:
        """Load sessions from file."""
        if os.path.exists(self.sessions_file):
            try:
                with open(self.sessions_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.sessions = [SessionEntry(**entry) for entry in data]
                    self._index = {s.id: s for s in self.sessions}
            except (IOError, json.JSONDecodeError):
                self.sessions = []
                self._index = {}
    
    def _save_sessions(self) -> None:
        """Save sessions to file."""
        try:
            with open(self.sessions_file, "w", encoding="utf-8") as f:
                data = [asdict(session) for session in self.sessions]
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except IOError:
            pass
    
    def create_session(
        self,
        name: str,
        documents: List[Dict[str, Any]],
        document_text: str,
        analysis_results: Dict[str, Any] = None
    ) -> SessionEntry:
        """Create a new session."""
        session = SessionEntry(
            id=f"{int(time.time())}_{hashlib.sha256(name.encode()).hexdigest()[:8]}",
            name=name,
            created_at=time.time(),
            updated_at=time.time(),
            documents=documents,
            chat_history=[],
            analysis_results=analysis_results or {},
            document_text=document_text,
            document_count=len(documents),
            word_count=len(document_text)
        )
        
        self.sessions.insert(0, session)
        self.sessions = self.sessions[:50]
        self._index = {s.id: s for s in self.sessions}
        self._save_sessions()
        return session
    
    def update_session(
        self,
        session_id: str,
        chat_history: List[Dict[str, str]] = None,
        analysis_results: Dict[str, Any] = None
    ) -> bool:
        """Update an existing session."""
        session = self.get_session_by_id(session_id)
        if session:
            if chat_history is not None:
                session.chat_history = chat_history
            if analysis_results is not None:
                session.analysis_results.update(analysis_results)
            session.updated_at = time.time()
            self._save_sessions()
            return True
        return False
    
    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """Add a message to session's chat history."""
        session = self.get_session_by_id(session_id)
        if session:
            session.chat_history.append({"role": role, "content": content})
            session.updated_at = time.time()
            self._save_sessions()
            return True
        return False
    
    def update_generated_content(self, session_id: str, key: str, value: Any) -> bool:
        """Persist generated content (summary/translation/report/compare) to session."""
        session = self.get_session_by_id(session_id)
        if session:
            session.generated_content[key] = value
            session.updated_at = time.time()
            self._save_sessions()
            return True
        return False
    
    def get_session_by_id(self, session_id: str) -> Optional[SessionEntry]:
        """Get session by ID (O(1) via dict index)."""
        return self._index.get(session_id)
    
    def get_all_sessions(self) -> List[SessionEntry]:
        """Get all sessions."""
        return self.sessions
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session by ID."""
        session = self.get_session_by_id(session_id)
        if session:
            self.sessions = [s for s in self.sessions if s.id != session_id]
            self._index = {s.id: s for s in self.sessions}
            self._save_sessions()
            return True
        return False
    
    def clear_sessions(self) -> None:
        """Clear all sessions."""
        self.sessions = []
        self._index = {}
        self._save_sessions()
    
    def search_by_name(self, query: str) -> List[SessionEntry]:
        """Search sessions by name."""
        query = query.lower()
        return [session for session in self.sessions if query in session.name.lower()]
    
    def get_recent_sessions(self, limit: int = 10) -> List[SessionEntry]:
        """Get most recent sessions."""
        return self.sessions[:limit]
