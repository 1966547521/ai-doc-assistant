"""Module for managing document history."""
import hashlib
import os
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

from src.runtime_paths import HISTORY_FILE, ensure_runtime_directories


@dataclass
class DocumentHistoryEntry:
    """Represents a document in history."""
    id: str
    filename: str
    file_path: str
    processed_at: float
    file_size: int
    word_count: int
    chunk_count: int
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    @property
    def processed_at_str(self) -> str:
        """Return formatted processing time."""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.processed_at))
    
    @property
    def file_size_human(self) -> str:
        """Return human-readable file size."""
        if self.file_size < 1024:
            return f"{self.file_size} B"
        elif self.file_size < 1024 * 1024:
            return f"{self.file_size / 1024:.1f} KB"
        else:
            return f"{self.file_size / (1024 * 1024):.1f} MB"


class HistoryManager:
    """Manages document processing history."""
    
    def __init__(self, history_file: str = str(HISTORY_FILE)):
        ensure_runtime_directories()
        self.history_file = history_file
        self.history: List[DocumentHistoryEntry] = []
        self._load_history()
    
    def _load_history(self) -> None:
        """Load history from file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.history = [DocumentHistoryEntry(**entry) for entry in data]
            except (IOError, json.JSONDecodeError):
                self.history = []
    
    def _save_history(self) -> None:
        """Save history to file."""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                data = [asdict(entry) for entry in self.history]
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except IOError:
            pass
    
    def add_entry(self, filename: str, file_path: str, file_size: int,
                  word_count: int, chunk_count: int, tags: List[str] = None,
                  entry_id: Optional[str] = None) -> None:
        """Add a new entry to history."""
        entry = DocumentHistoryEntry(
            id=entry_id or f"{int(time.time())}_{hashlib.sha256(filename.encode()).hexdigest()[:8]}",
            filename=filename,
            file_path=file_path,
            processed_at=time.time(),
            file_size=file_size,
            word_count=word_count,
            chunk_count=chunk_count,
            tags=tags or []
        )
        
        # Remove duplicate entries for the same file
        self.history = [h for h in self.history if h.filename != filename]
        
        # Add new entry at the beginning
        self.history.insert(0, entry)
        
        # Keep only last 50 entries
        self.history = self.history[:50]
        
        self._save_history()
    
    def get_all_entries(self) -> List[DocumentHistoryEntry]:
        """Get all history entries."""
        return self.history
    
    def get_entry_by_id(self, entry_id: str) -> Optional[DocumentHistoryEntry]:
        """Get entry by ID."""
        for entry in self.history:
            if entry.id == entry_id:
                return entry
        return None
    
    def delete_entry(self, entry_id: str) -> bool:
        """Delete entry by ID."""
        entry = self.get_entry_by_id(entry_id)
        if entry:
            self.history = [h for h in self.history if h.id != entry_id]
            self._save_history()
            return True
        return False
    
    def update_entry_tags(self, entry_id: str, tags: List[str]) -> bool:
        """Update tags for an entry."""
        entry = self.get_entry_by_id(entry_id)
        if entry:
            entry.tags = tags
            self._save_history()
            return True
        return False
    
    def clear_history(self) -> None:
        """Clear all history entries."""
        self.history = []
        self._save_history()
    
    def search_by_name(self, query: str) -> List[DocumentHistoryEntry]:
        """Search entries by filename."""
        query = query.lower()
        return [entry for entry in self.history if query in entry.filename.lower()]
    
    def get_recent_entries(self, limit: int = 10) -> List[DocumentHistoryEntry]:
        """Get most recent entries."""
        return self.history[:limit]
