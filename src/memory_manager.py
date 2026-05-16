"""Memory management for conversational context."""

from typing import Dict, List


class MemoryManager:
    """Manages conversation history for multi-turn dialogues."""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-self.max_history * 2 :]

    def get_history(self) -> str:
        """Get formatted conversation history."""
        history = ""
        for msg in self.messages:
            if msg["role"] == "user":
                history += f"用户: {msg['content']}\n"
            elif msg["role"] == "assistant":
                history += f"助手: {msg['content']}\n"
        return history.strip()

    def get_messages(self) -> List[Dict]:
        """Get conversation history as a list of dicts."""
        return self.messages.copy()

    def clear_history(self):
        """Clear all conversation history."""
        self.messages = []

    def get_history_length(self) -> int:
        """Get the number of messages in history."""
        return len(self.messages)
