"""Document structure analysis with regex extraction + LLM-powered deep understanding.

Two-phase approach:
  Phase 1 — Regex: fast heading/section extraction (always works, no API call)
  Phase 2 — LLM: document-type recognition, section-content understanding,
           structure overview, and quality assessment.
"""

import re
from typing import Dict, List, Tuple, Optional

from src.logger import get_logger
from src.llm_enhancer import LLMEnhancer

logger = get_logger(__name__)


class StructureAnalyzer:
    """Analyzes document structure — regex extraction + LLM semantic understanding."""

    def __init__(self):
        logger.info("Initializing StructureAnalyzer")

        self.title_patterns = [
            (r"^#{1}\s+(.+)", "h1"),
            (r"^#{2}\s+(.+)", "h2"),
            (r"^#{3}\s+(.+)", "h3"),
            (r"^#{4}\s+(.+)", "h4"),
            (r"^【(.+)】", "h2"),
            (r"^\d+\.\s+(.+)", "h2"),
            (r"^\d+\.\d+\s+(.+)", "h3"),
            (r"^\d+\.\d+\.\d+\s+(.+)", "h4"),
            (r"^(.+)[:：]", "h3"),
        ]

        self._llm_enhancer: Optional[LLMEnhancer] = None

        self.generic_structure = [
            "引言", "背景", "概述", "简介",
            "方法", "材料", "实验",
            "结果", "分析", "讨论",
            "结论", "总结", "展望",
            "参考文献", "附录", "致谢"
        ]

        self._llm_available: Optional[bool] = None
        logger.debug("StructureAnalyzer initialized with %d title patterns", len(self.title_patterns))

    # ── LLM helper ──────────────────────────────────────────────

    def _get_enhancer(self) -> Optional[LLMEnhancer]:
        if self._llm_enhancer is None:
            try:
                from src.llm_enhancer import LLMEnhancer
                from src.utils import get_llm
                self._llm_enhancer = LLMEnhancer(get_llm())
                self._llm_available = True
                logger.debug("LLM enhancer initialized for structure analysis")
            except Exception as e:
                logger.warning("Failed to initialize LLM enhancer: %s", str(e))
                self._llm_available = False
                return None
        return self._llm_enhancer

    def _llm_is_available(self) -> bool:
        if self._llm_available is None:
            self._get_enhancer()
        return self._llm_available or False

    # ── Heading validation ──────────────────────────────────────

    def _is_valid_heading(self, text: str) -> bool:
        if not text:
            return False
        if len(text) > 100:
            return False
        forbidden = [
            r"https?://", r"http://", r"\.com", r"\.org", r"\.net",
            r"@\w+", r"\.\w+:", r"^\s*def\s+", r"^\s*class\s+", r"^\s*import\s+", r"^\s*from\s+",
        ]
        for pattern in forbidden:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        if "/" in text and len(text) > 10 and re.search(r"\w+/\w+", text):
            if re.search(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", text):
                return False
        return True

    # ── Phase 1: Regex extraction ───────────────────────────────

    def extract_headings(self, text: str, use_llm: bool = True) -> List[Dict[str, str]]:
        logger.info("Extracting headings (regex phase)")
        lines = text.split("\n")
        headings = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            for pattern, level in self.title_patterns:
                match = re.match(pattern, line)
                if match:
                    heading_text = match.group(1).strip()
                    if self._is_valid_heading(heading_text):
                        headings.append({"text": heading_text, "level": level, "line_number": i + 1})
                    break
        logger.info("Extracted %d headings", len(headings))
        return headings

    def calculate_structure_similarity(self, headings: List[Dict[str, str]]) -> float:
        if not headings:
            return 0.0
        heading_texts = [h["text"] for h in headings]
        matched = sum(1 for h in heading_texts for g in self.generic_structure if g in h or h in g)
        return matched / len(self.generic_structure)

    def build_document_tree(self, headings: List[Dict[str, str]]) -> list:
        tree: list = []
        stack: list = []
        level_order = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}
        for heading in headings:
            current_level = level_order.get(heading["level"], 5)
            node = {
                "text": heading["text"], "level": heading["level"],
                "children": [], "line_number": heading["line_number"],
            }
            while stack and level_order.get(stack[-1]["level"], 5) >= current_level:
                stack.pop()
            if stack:
                stack[-1]["children"].append(node)
            else:
                tree.append(node)
            stack.append(node)
        return tree

    def extract_sections(self, text: str) -> Tuple[List[Dict[str, str | int]], str]:
        lines = text.split("\n")
        sections: List[Dict[str, str | int]] = []
        current_section: Optional[Dict] = None
        current_content: List[str] = []
        preamble_lines: List[str] = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            is_heading = False
            matched_text = ""
            matched_level = "h2"

            for pattern, level in self.title_patterns:
                match = re.match(pattern, stripped)
                if match:
                    ht = match.group(1).strip()
                    if self._is_valid_heading(ht):
                        is_heading, matched_text, matched_level = True, ht, level
                        break

            if is_heading:
                if current_section:
                    current_section["content"] = "\n".join(current_content).strip()
                    sections.append(current_section)
                current_section = {"title": matched_text, "level": matched_level, "start_line": i + 1, "content": ""}
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
                else:
                    preamble_lines.append(line)

        if current_section:
            current_section["content"] = "\n".join(current_content).strip()
            sections.append(current_section)

        preamble = "\n".join(preamble_lines).strip()
        logger.info("Extracted %d sections, %d preamble chars", len(sections), len(preamble))
        return sections, preamble

    def validate_heading_levels(self, headings: List[Dict[str, str]]) -> dict:
        level_order = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}
        level_jumps, duplicate_headings = [], []

        if not headings:
            return {"has_issues": False, "summary": "", "suggestions": [], "details": []}

        prev_level = level_order.get(headings[0]["level"], 5)
        for i, heading in enumerate(headings[1:], 1):
            cur = level_order.get(heading["level"], 5)
            if cur > prev_level + 1:
                level_jumps.append({"line": i + 1, "current_level": heading["level"],
                                     "prev_level": headings[i]["level"], "title": heading["text"]})
            if cur == prev_level and headings[i]["text"] == heading["text"]:
                duplicate_headings.append({"line": i + 1, "title": heading["text"]})
            prev_level = cur

        summary_parts, suggestions = [], []
        if duplicate_headings:
            summary_parts.append(f"检测到 {len(duplicate_headings)} 处重复标题")
            suggestions.append("建议检查并修改重复的标题，确保每个标题唯一")
        if level_jumps:
            summary_parts.append(f"检测到 {len(level_jumps)} 处标题层级跳跃")
            suggestions.append("建议按照 h1 > h2 > h3 > h4 的层级顺序组织标题")

        summary = "文档结构存在以下问题：" + "；".join(summary_parts) + "。" if summary_parts else "文档标题层级结构良好，未检测到明显问题。"
        details = []
        for d in duplicate_headings:
            details.append(f"第 {d['line']} 行：重复的标题 '{d['title']}'")
        for j in level_jumps:
            details.append(f"第 {j['line']} 行：标题层级跳跃 ({j['current_level']} 跟在 {j['prev_level']} 之后)")

        return {"has_issues": len(summary_parts) > 0, "summary": summary, "suggestions": suggestions, "details": details}

    def calculate_section_stats(self, sections: List[Dict[str, str | int]]) -> dict:
        if not sections:
            return {"total_sections": 0, "avg_length": 0, "max_length": 0, "min_length": 0,
                    "longest_section": None, "shortest_section": None, "total_chars": 0}
        lengths = [len(s.get("content", "") or "") for s in sections]
        total = sum(lengths)
        max_idx = lengths.index(max(lengths))
        min_idx = lengths.index(min(lengths))
        return {
            "total_sections": len(sections), "avg_length": round(total / len(sections), 2),
            "max_length": max(lengths), "min_length": min(lengths),
            "longest_section": sections[max_idx]["title"], "shortest_section": sections[min_idx]["title"],
            "total_chars": total,
        }

    # ── Phase 2: LLM-powered deep analysis ───────────────────────

    def _llm_deep_analysis(self, text: str, sections: List[Dict[str, str | int]]) -> dict:
        """Use LLM to understand document type, section content, and structure quality.

        Returns dict with keys: doc_type, doc_purpose, section_summaries, overview, quality
        """
        if not self._llm_is_available():
            return {}

        enhancer = self._get_enhancer()
        if not enhancer:
            return {}

        # Build a compact representation: section title + first 300 chars of content
        section_snippets = []
        for i, sec in enumerate(sections, 1):
            title = sec.get("title", f"Section {i}")
            content = (sec.get("content", "") or "")[:300]
            section_snippets.append(f"§{i} 【{title}】\n{content}")

        snippets_text = "\n\n".join(section_snippets)
        if len(snippets_text) > 6000:
            snippets_text = snippets_text[:6000] + "\n...(truncated)"

        prompt = f"""你是一位专业的文档分析师。请分析以下文档的结构，不要只复述标题，而要理解内容。

【文档节选】
{snippets_text}

请用 JSON 格式回复（不要markdown代码块，直接JSON）：
{{
  "doc_type": "文档类型（如：学术论文/技术报告/商业计划书/会议纪要/新闻稿/产品手册/其他）",
  "doc_purpose": "一句话概括文档的核心目的",
  "section_summaries": [
    {{"title": "原标题", "summary": "基于该节内容的一句话总结，不是重复标题"}}
  ],
  "overview": "一段话（2-3句）概述文档的组织逻辑和行文脉络",
  "quality": {{
    "level": "high/medium/low",
    "strengths": ["优点1"],
    "weaknesses": ["不足1"],
    "suggestions": ["改进建议1"]
  }}
}}
<END>"""

        try:
            response = enhancer.llm.invoke(prompt)
            result = enhancer._parse_json_response(response.content)
            if result:
                logger.info("LLM deep analysis completed")
                return result
        except Exception as e:
            logger.error("LLM deep analysis failed: %s", str(e))

        return {}

    def _summarize_sections_batch(self, sections: List[Dict[str, str | int]]) -> Dict[str, str]:
        """Lightweight: generate one-sentence summaries for each section via LLM."""
        if not self._llm_is_available() or len(sections) <= 1:
            return {}

        enhancer = self._get_enhancer()
        if not enhancer:
            return {}

        items = []
        for i, sec in enumerate(sections):
            title = sec.get("title", "")
            content = (sec.get("content", "") or "")[:200]
            items.append(f"[{i}] 标题: {title}\n内容: {content}")

        prompt = f"""为以下每个章节写一句话总结（基于内容而非标题）：

{chr(10).join(items[:10])}

请用 JSON 格式回复，key 是章节编号，value 是总结：
{{"0": "总结0", "1": "总结1", ...}}
<END>"""

        try:
            response = enhancer.llm.invoke(prompt)
            result = enhancer._parse_json_response(response.content)
            if result and isinstance(result, dict):
                # Map back to section titles
                mapped = {}
                for key_str, summary in result.items():
                    try:
                        idx = int(key_str)
                        if 0 <= idx < len(sections):
                            mapped[sections[idx].get("title", "")] = summary
                    except (ValueError, TypeError):
                        pass
                return mapped
        except Exception as e:
            logger.error("Section summarization failed: %s", str(e))

        return {}

    # ── Main analysis orchestrator ───────────────────────────────

    def analyze_document(self, text: str, similarity_threshold: float = 0.1, use_llm: bool = True) -> dict:
        """Two-phase analysis: regex extraction + LLM semantic understanding.

        Returns a dict with all structure information plus optional LLM insights:
          doc_type, doc_purpose, section_summaries, overview, quality
        """
        logger.info("Starting document analysis (use_llm=%s)", use_llm)

        headings = self.extract_headings(text)
        similarity = self.calculate_structure_similarity(headings)
        has_structure = len(headings) >= 2 or similarity >= similarity_threshold

        if not has_structure and len(headings) < 2:
            logger.info("Document has no recognizable structure")
            return {
                "headings": [], "structure_tree": [], "sections": [], "preamble": text,
                "total_headings": 0, "total_sections": 0, "depth": 0,
                "section_stats": self.calculate_section_stats([]),
                "has_preamble": True, "has_structure": False,
                "structure_similarity": 0.0, "message": "本文无明显结构",
                "validation": {"has_issues": False, "summary": "", "suggestions": [], "details": []},
            }

        tree = self.build_document_tree(headings)
        sections, preamble = self.extract_sections(text)
        validation = self.validate_heading_levels(headings)
        stats = self.calculate_section_stats(sections)

        result = {
            "headings": headings, "structure_tree": tree, "sections": sections,
            "preamble": preamble, "total_headings": len(headings), "total_sections": len(sections),
            "depth": self._calculate_depth(tree), "validation": validation, "section_stats": stats,
            "has_preamble": bool(preamble), "has_structure": has_structure,
            "structure_similarity": similarity, "message": None,
        }

        # Phase 2: LLM-powered deep analysis
        if use_llm and sections:
            llm_insight = self._llm_deep_analysis(text, sections)
            if llm_insight:
                result["doc_type"] = llm_insight.get("doc_type", "未知")
                result["doc_purpose"] = llm_insight.get("doc_purpose", "")
                result["section_summaries"] = llm_insight.get("section_summaries", [])
                result["overview"] = llm_insight.get("overview", "")
                result["quality"] = llm_insight.get("quality", {})
            else:
                # Fallback: lightweight section summarization
                summaries = self._summarize_sections_batch(sections)
                if summaries:
                    result["section_summaries"] = [
                        {"title": title, "summary": summary}
                        for title, summary in summaries.items()
                    ]

        logger.info("Document analysis completed: %d headings, %d sections", len(headings), len(sections))
        return result

    # ── Tree utilities ──────────────────────────────────────────

    def _calculate_depth(self, tree: List[Dict], current_depth: int = 1) -> int:
        max_depth = current_depth
        for node in tree:
            if node["children"]:
                child = self._calculate_depth(node["children"], current_depth + 1)
                max_depth = max(max_depth, child)
        return max_depth

    def format_tree(self, tree: List[Dict], indent: int = 0) -> str:
        result = []
        for node in tree:
            result.append("  " * indent + f"- {node['text']}")
            if node["children"]:
                result.append(self.format_tree(node["children"], indent + 1))
        return "\n".join(result)

    def format_tree_outline(self, tree: List[Dict], parent_num: str = "", level: int = 1) -> List[Dict]:
        chinese_nums = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        result = []
        for idx, node in enumerate(tree, 1):
            if level == 1:
                num = chinese_nums[min(idx, len(chinese_nums) - 1)]
                display_num = f"{num}、"
            else:
                display_num = f"{parent_num}.{idx}" if parent_num else str(idx)

            item = {
                "text": node.get("text", ""), "number": display_num, "level": level,
                "has_children": bool(node.get("children", [])), "children": [],
                "content_preview": node.get("content_preview", ""),
            }
            if node.get("children"):
                item["children"] = self.format_tree_outline(
                    node["children"],
                    str(idx) if not parent_num or level == 1 else f"{parent_num}.{idx}",
                    level + 1,
                )
            result.append(item)
        return result

    # ── TOC generators ──────────────────────────────────────────

    def generate_toc(self, sections: List[Dict[str, str | int]], format_type: str = "markdown") -> str:
        if not sections:
            return "无章节"
        if format_type == "markdown":
            return self._generate_markdown_toc(sections)
        elif format_type == "text":
            return self._generate_text_toc(sections)
        elif format_type == "html":
            return self._generate_html_toc(sections)
        else:
            return self._generate_markdown_toc(sections)

    def _generate_markdown_toc(self, sections):
        lines = ["# 目录\n"]
        prefix = {"h1": "", "h2": "  ", "h3": "    ", "h4": "      "}
        for i, sec in enumerate(sections, 1):
            lines.append(f"{prefix.get(sec.get('level', 'h2'), '  ')}{i}. {sec.get('title', '')}")
        return "\n".join(lines)

    def _generate_text_toc(self, sections):
        return "\n".join(["目录", "=" * 40] + [f"{i}. {s.get('title', '')}" for i, s in enumerate(sections, 1)])

    def _generate_html_toc(self, sections):
        return "<h2>目录</h2>\n<ul>\n" + "\n".join(
            f"  <li>{i}. {s.get('title', '')}</li>" for i, s in enumerate(sections, 1)
        ) + "\n</ul>"

    def get_level_distribution(self, headings: List[Dict[str, str]]) -> Dict[str, int]:
        dist = {"h1": 0, "h2": 0, "h3": 0, "h4": 0}
        for h in headings:
            level = h.get("level", "h2")
            if level in dist:
                dist[level] += 1
        return dist
