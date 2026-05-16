"""Document translation engine with streaming support."""

from typing import Iterator, Dict, Optional

from langchain_core.language_models import BaseChatModel

from src.prompt_manager import prompt_manager
from src.utils import get_llm


class TranslationEngine:
    """Handles document translation with LLM and quality checking."""

    SUPPORTED_LANGUAGES = [
        {"code": "zh", "name": "中文"},
        {"code": "en", "name": "English"},
        {"code": "ja", "name": "日本語"},
        {"code": "ko", "name": "한국어"},
        {"code": "fr", "name": "Français"},
        {"code": "de", "name": "Deutsch"},
        {"code": "es", "name": "Español"},
        {"code": "ru", "name": "Русский"},
    ]

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm if llm is not None else get_llm()
        self._enhancer = None

    def _get_enhancer(self):
        """延迟初始化LLM增强器"""
        if self._enhancer is None:
            from .llm_enhancer import LLMEnhancer
            self._enhancer = LLMEnhancer(self.llm)
        return self._enhancer

    def translate(
        self, text: str, target_lang: str = "en", source_lang: str = "zh", check_quality: bool = False
    ) -> Dict[str, str] | str:
        """Translate text to target language.
        
        Args:
            text: The text to translate
            target_lang: Target language code
            source_lang: Source language code
            check_quality: Whether to check translation quality with LLM
            
        Returns:
            If check_quality is True, returns dict with 'translation' and 'quality' keys
            Otherwise returns just the translation string
        """
        target_name = self._get_lang_name(target_lang)
        source_name = self._get_lang_name(source_lang)

        prompt = prompt_manager.get_prompt(
            "translate",
            f"""请将以下{source_name}文本翻译成{target_name}：

{text}

请保持原文的格式和结构，不要添加额外内容。""",
        )
        
        # Check if prompt has placeholders, otherwise append text
        if "{text}" in prompt and "{target_language}" in prompt:
            prompt = prompt.format(text=text[:5000], target_language=target_name)
        elif "{text}" in prompt:
            prompt = prompt.format(text=text[:5000])
        else:
            prompt = prompt + "\n\n【待翻译文本】\n" + text[:5000] + "\n\n【目标语言】\n" + target_name

        response = self.llm.invoke(prompt)
        translation = response.content
        
        # LLM检查翻译质量
        if check_quality:
            enhancer = self._get_enhancer()
            quality = enhancer.check_translation_quality(text[:1000], translation, target_name)
            return {"translation": translation, "quality": quality}
        
        return translation

    def translate_chunked(
        self, text: str, target_lang: str = "en", chunk_size: int = 2000, check_quality: bool = False
    ) -> Dict[str, str] | str:
        """Translate long text in chunks and combine results.
        
        Args:
            text: The text to translate
            target_lang: Target language code
            chunk_size: Size of each chunk
            check_quality: Whether to check translation quality with LLM
            
        Returns:
            If check_quality is True, returns dict with 'translation' and 'quality' keys
            Otherwise returns just the translation string
        """
        target_name = self._get_lang_name(target_lang)

        # Split text into chunks
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i : i + chunk_size])

        results = []
        for i, chunk in enumerate(chunks):
            prompt = prompt_manager.get_prompt(
                "translate",
                f"""请将以下文本翻译成{target_name}（第{i+1}/{len(chunks)}部分）：

{chunk}

请保持原文格式，不要添加额外内容。""",
            )
            
            # Check if prompt has placeholders, otherwise append text
            if "{text}" in prompt and "{target_language}" in prompt:
                prompt = prompt.format(text=chunk[:5000], target_language=target_name)
            elif "{text}" in prompt:
                prompt = prompt.format(text=chunk[:5000])
            else:
                prompt = prompt + "\n\n【待翻译文本】\n" + chunk[:5000] + "\n\n【目标语言】\n" + target_name
            
            response = self.llm.invoke(prompt)
            results.append(response.content)

        translation = "\n".join(results)
        
        # LLM检查翻译质量
        if check_quality:
            enhancer = self._get_enhancer()
            quality = enhancer.check_translation_quality(text[:1000], translation, target_name)
            return {"translation": translation, "quality": quality}
        
        return translation

    def stream_translate(
        self, text: str, target_lang: str = "en", source_lang: str = "zh"
    ) -> Iterator[str]:
        """Stream translation result. Auto-chunks text longer than 12000 chars."""
        MAX_CHARS = 12000
        if len(text) <= MAX_CHARS:
            yield from self._stream_single(text, target_lang, source_lang)
        else:
            yield from self.stream_translate_chunked(text, target_lang, source_lang)

    def _stream_single(
        self, text: str, target_lang: str = "en", source_lang: str = "zh"
    ) -> Iterator[str]:
        """Stream translation for text within the char limit."""
        target_name = self._get_lang_name(target_lang)
        source_name = self._get_lang_name(source_lang)

        prompt = prompt_manager.get_prompt(
            "translate",
            f"""请将以下{source_name}文本翻译成{target_name}：

{text}

要求：
1. 保持原文的段落结构和格式，包括空行、缩进
2. 保留所有标点符号、数字、百分比、单位符号
3. 保留所有特殊字符，如 ( ) [ ] {{ }} < > / \\ | @ # $ % ^ & * + - = _ 等
4. 保留所有日期、时间、金额的原始格式
5. 不要添加额外内容或解释""",
        )
        
        if "{text}" in prompt and "{target_language}" in prompt:
            prompt = prompt.format(text=text, target_language=target_name)
        elif "{text}" in prompt:
            prompt = prompt.format(text=text)
        else:
            prompt = prompt + "\n\n【待翻译文本】\n" + text + "\n\n【目标语言】\n" + target_name

        try:
            for chunk in self.llm.stream(prompt):
                yield chunk.content
        except (ConnectionError, RuntimeError) as e:
            yield f"翻译过程中出现错误: {str(e)}"

    def stream_translate_chunked(
        self, text: str, target_lang: str = "en", source_lang: str = "zh",
        chunk_size: int = 4000
    ) -> Iterator[str]:
        """Stream translation for long texts, split into chunks with clean output."""
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        total = len(chunks)

        if total > 1:
            yield f"⏳ 全文共 {total} 段，分段翻译中...\n\n"

        for idx, chunk in enumerate(chunks, 1):
            if total > 1:
                yield f"⏳ 翻译进度: [{('█' * idx).ljust(total, '░')}] {idx}/{total}\n\n"
            yield from self._stream_single(chunk, target_lang, source_lang)
            yield "\n\n"

        yield f"✅ 翻译完成（共 {total} 段）\n\n"

    def detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        prompt = """请识别以下文本的语言，只需输出语言名称（中文、English、日本語等）：

{text}"""

        response = self.llm.invoke(prompt.format(text=text[:500]))
        return response.content.strip()

    @staticmethod
    def _get_lang_name(code: str) -> str:
        """Get language name from code."""
        lang_map = {
            "zh": "中文",
            "en": "English",
            "ja": "日本語",
            "ko": "한국어",
            "fr": "Français",
            "de": "Deutsch",
            "es": "Español",
            "ru": "Русский",
        }
        return lang_map.get(code, code)

    @staticmethod
    def get_language_options():
        """Get list of supported languages for UI."""
        return [
            (lang["code"], lang["name"])
            for lang in TranslationEngine.SUPPORTED_LANGUAGES
        ]
