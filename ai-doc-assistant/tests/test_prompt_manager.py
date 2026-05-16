"""Tests for PromptManager functionality."""
import os
import tempfile
import pytest
from src.prompt_manager import PromptManager


class TestPromptManager:
    """Test cases for PromptManager."""
    
    @pytest.fixture
    def temp_prompts_dir(self):
        """Create a temporary prompts directory."""
        temp_dir = tempfile.mkdtemp(prefix="test_prompts_")
        yield temp_dir
        if os.path.exists(temp_dir):
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
            os.rmdir(temp_dir)
    
    @pytest.fixture
    def prompt_manager(self, temp_prompts_dir):
        """Create a PromptManager with temporary directory."""
        # Create a test prompt file
        prompt_path = os.path.join(temp_prompts_dir, "test_prompt.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write("你好 {name}，欢迎使用 {product}！")
        
        return PromptManager(prompts_dir=temp_prompts_dir)
    
    @pytest.fixture
    def empty_prompt_manager(self, temp_prompts_dir):
        """Create a PromptManager with empty directory."""
        return PromptManager(prompts_dir=temp_prompts_dir)
    
    def test_load_prompts(self, prompt_manager):
        """Test loading prompts from directory."""
        assert prompt_manager.has_prompt("test_prompt")
        content = prompt_manager.get_prompt("test_prompt")
        assert "你好" in content
        assert "{name}" in content
    
    def test_get_prompt_existing(self, prompt_manager):
        """Test getting an existing prompt."""
        content = prompt_manager.get_prompt("test_prompt")
        assert content is not None
        assert "你好" in content
    
    def test_get_prompt_nonexistent(self, prompt_manager):
        """Test getting a non-existent prompt."""
        content = prompt_manager.get_prompt("nonexistent")
        assert content is None
    
    def test_get_prompt_with_default(self, prompt_manager):
        """Test getting a prompt with default value."""
        content = prompt_manager.get_prompt("nonexistent", "default content")
        assert content == "default content"
    
    def test_has_prompt_true(self, prompt_manager):
        """Test checking if a prompt exists."""
        assert prompt_manager.has_prompt("test_prompt")
    
    def test_has_prompt_false(self, prompt_manager):
        """Test checking if a prompt doesn't exist."""
        assert not prompt_manager.has_prompt("nonexistent")
    
    def test_set_prompt(self, prompt_manager):
        """Test setting a new prompt."""
        prompt_manager.set_prompt("new_prompt", "新提示词内容")
        
        assert prompt_manager.has_prompt("new_prompt")
        content = prompt_manager.get_prompt("new_prompt")
        assert content == "新提示词内容"
        
        # Verify it was saved to file
        prompt_path = os.path.join(prompt_manager.prompts_dir, "new_prompt.txt")
        assert os.path.exists(prompt_path)
    
    def test_list_prompts(self, prompt_manager):
        """Test listing all prompts."""
        prompts = prompt_manager.list_prompts()
        assert "test_prompt" in prompts
        assert isinstance(prompts, list)
    
    def test_render_prompt(self, prompt_manager):
        """Test rendering a prompt with variables."""
        result = prompt_manager.render_prompt(
            "test_prompt", name="张三", product="AI助手"
        )
        assert "张三" in result
        assert "AI助手" in result
    
    def test_render_prompt_not_found(self, prompt_manager):
        """Test rendering a non-existent prompt raises error."""
        with pytest.raises(ValueError, match="not found"):
            prompt_manager.render_prompt("nonexistent", key="value")
    
    def test_reload_prompts(self, prompt_manager, temp_prompts_dir):
        """Test reloading prompts from files."""
        original_content = prompt_manager.get_prompt("test_prompt")
        
        # Modify the file directly
        prompt_path = os.path.join(temp_prompts_dir, "test_prompt.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write("Updated content")
        
        prompt_manager.reload()
        updated_content = prompt_manager.get_prompt("test_prompt")
        assert updated_content == "Updated content"
        assert updated_content != original_content
    
    def test_load_from_nonexistent_directory(self):
        """Test creating manager with non-existent directory."""
        mgr = PromptManager(prompts_dir="/nonexistent/path/prompts")
        assert mgr.list_prompts() == []
