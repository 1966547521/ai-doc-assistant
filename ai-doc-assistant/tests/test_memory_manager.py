"""Tests for MemoryManager functionality."""
import pytest
from src.memory_manager import MemoryManager


class TestMemoryManager:
    """Test cases for MemoryManager."""
    
    @pytest.fixture
    def memory(self):
        """Create a MemoryManager instance."""
        return MemoryManager(max_history=5)
    
    def test_add_message(self, memory):
        """Test adding a message."""
        memory.add_message("user", "你好")
        
        assert memory.get_history_length() == 1
        messages = memory.get_messages()
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "你好"
    
    def test_add_multiple_messages(self, memory):
        """Test adding multiple messages."""
        memory.add_message("user", "问题1")
        memory.add_message("assistant", "回答1")
        memory.add_message("user", "问题2")
        memory.add_message("assistant", "回答2")
        
        assert memory.get_history_length() == 4
    
    def test_get_history(self, memory):
        """Test getting formatted history."""
        memory.add_message("user", "你好")
        memory.add_message("assistant", "你好！有什么可以帮助你的？")
        
        history = memory.get_history()
        
        assert "用户: 你好" in history
        assert "助手: 你好！有什么可以帮助你的？" in history
    
    def test_get_history_empty(self, memory):
        """Test getting history when empty."""
        history = memory.get_history()
        assert history == ""
    
    def test_get_messages_copy(self, memory):
        """Test that get_messages returns a copy."""
        memory.add_message("user", "test")
        
        messages = memory.get_messages()
        messages.append({"role": "assistant", "content": "modified"})
        
        assert memory.get_history_length() == 1
    
    def test_max_history_limit(self, memory):
        """Test that history size is limited."""
        # Add more messages than max_history * 2
        for i in range(15):
            memory.add_message("user" if i % 2 == 0 else "assistant", f"msg{i}")
        
        assert len(memory.messages) <= 10  # max_history * 2
    
    def test_clear_history(self, memory):
        """Test clearing history."""
        memory.add_message("user", "test")
        memory.add_message("assistant", "response")
        
        memory.clear_history()
        
        assert memory.get_history_length() == 0
        assert memory.get_history() == ""
    
    def test_get_history_length(self, memory):
        """Test getting history length."""
        assert memory.get_history_length() == 0
        
        memory.add_message("user", "test")
        assert memory.get_history_length() == 1
    
    def test_custom_max_history(self):
        """Test custom max_history setting."""
        memory = MemoryManager(max_history=3)
        
        for i in range(10):
            memory.add_message("user", f"msg{i}")
        
        assert len(memory.messages) <= 6
