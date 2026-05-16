"""Prompt manager for loading and managing custom prompts."""

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages prompt templates loaded from files."""

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = prompts_dir
        self.prompts: Dict[str, str] = {}
        self._load_prompts()

    def _load_prompts(self):
        """Load all prompt files from the prompts directory."""
        if not os.path.exists(self.prompts_dir):
            os.makedirs(self.prompts_dir, exist_ok=True)
            return

        for filename in os.listdir(self.prompts_dir):
            if filename.endswith(".txt"):
                prompt_name = filename[:-4]  # Remove .txt extension
                filepath = os.path.join(self.prompts_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        self.prompts[prompt_name] = f.read()
                except IOError as e:
                    logger.warning("Error loading prompt %s: %s", filename, e)

    def get_prompt(self, prompt_name: str, default: Optional[str] = None) -> Optional[str]:
        """Get a prompt by name."""
        return self.prompts.get(prompt_name, default)

    def has_prompt(self, prompt_name: str) -> bool:
        """Check if a prompt exists."""
        return prompt_name in self.prompts

    def set_prompt(self, prompt_name: str, content: str):
        """Set a prompt (saves to file)."""
        self.prompts[prompt_name] = content
        filepath = os.path.join(self.prompts_dir, f"{prompt_name}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    def list_prompts(self) -> list:
        """List all available prompts."""
        return list(self.prompts.keys())

    def render_prompt(self, prompt_name: str, **kwargs) -> str:
        """Render a prompt with variables."""
        prompt = self.get_prompt(prompt_name)
        if not prompt:
            raise ValueError(f"Prompt '{prompt_name}' not found")
        try:
            return prompt.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable in prompt '{prompt_name}': {e}") from e

    def reload(self):
        """Reload prompts from files."""
        self.prompts.clear()
        self._load_prompts()


# Global prompt manager instance
prompt_manager = PromptManager()
