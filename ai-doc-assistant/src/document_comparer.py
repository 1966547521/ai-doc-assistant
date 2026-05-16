"""Document comparison module for detecting differences and similarities."""

from difflib import Differ, SequenceMatcher
from typing import Dict, List, Optional, Tuple

from langchain_core.language_models import BaseChatModel

from src.prompt_manager import prompt_manager
from src.utils import get_llm


class DocumentComparer:
    """Compares two documents and highlights differences."""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm if llm is not None else get_llm()

    def compare_texts(self, text1: str, text2: str) -> Dict[str, List[str] | float | Dict[str, int]]:
        """Compare two texts and return differences."""
        differ = Differ()
        diff = list(differ.compare(text1.splitlines(), text2.splitlines()))

        added_lines: List[str] = []
        removed_lines: List[str] = []
        changed_lines: List[str] = []
        unchanged_lines: List[str] = []

        for line in diff:
            if line.startswith("+ "):
                added_lines.append(line[2:])
            elif line.startswith("- "):
                removed_lines.append(line[2:])
            elif line.startswith("? "):
                pass  # Skip change indicators
            elif line.startswith("  "):
                unchanged_lines.append(line[2:])

        result: Dict[str, List[str] | float | Dict[str, int]] = {
            "added": added_lines,
            "removed": removed_lines,
            "changed": changed_lines,
            "unchanged": unchanged_lines,
            "similarity": self.calculate_similarity(text1, text2),
            "stats": self._calculate_stats(diff),
        }

        return result

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using SequenceMatcher."""
        matcher = SequenceMatcher(None, text1, text2)
        return round(matcher.ratio() * 100, 2)

    def _calculate_stats(self, diff: List[str]) -> Dict:
        """Calculate statistics from diff result."""
        stats = {
            "added_lines": 0,
            "removed_lines": 0,
            "unchanged_lines": 0,
            "total_lines": len([d for d in diff if not d.startswith("? ")]),
        }

        for line in diff:
            if line.startswith("+ "):
                stats["added_lines"] += 1
            elif line.startswith("- "):
                stats["removed_lines"] += 1
            elif line.startswith("  "):
                stats["unchanged_lines"] += 1

        return stats

    def generate_diff_summary(
        self, text1: str, text2: str, title1: str = "文档A", title2: str = "文档B"
    ) -> str:
        """Generate a summary of differences in natural language."""
        diff_result = self.compare_texts(text1, text2)

        prompt = prompt_manager.get_prompt(
            "compare",
            f"""请分析以下两个文档的差异并给出总结：

【{title1}】（旧版本）
{text1[:3000]}

【{title2}】（新版本）
{text2[:3000]}

请按照以下结构输出：
1. 整体相似度：{diff_result['similarity']}%
2. 主要变化：
   - 新增内容：列出新增的主要内容
   - 删除内容：列出删除的主要内容
   - 修改内容：列出修改的主要内容
3. 变化影响评估：分析这些变化可能带来的影响

请用简洁明了的语言描述，不要使用技术术语。""",
        )

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception:
            return f"相似度: {diff_result['similarity']}%。新增 {diff_result['stats']['added_lines']} 行，删除 {diff_result['stats']['removed_lines']} 行。"

    def highlight_differences(self, text1: str, text2: str) -> Tuple[str, str]:
        """Highlight differences in HTML format for visualization."""
        differ = Differ()
        diff = list(differ.compare(text1.splitlines(), text2.splitlines()))

        html1_lines = []
        html2_lines = []

        i = 0
        while i < len(diff):
            line = diff[i]

            if line.startswith("- "):
                # Removed from text1
                html1_lines.append(
                    f'<span style="background-color: #ffcccc; text-decoration: line-through;">{line[2:]}</span>'
                )

                # Check if next line is added
                if i + 1 < len(diff) and diff[i + 1].startswith("+ "):
                    html2_lines.append(
                        f'<span style="background-color: #ccffcc;">{diff[i + 1][2:]}</span>'
                    )
                    i += 1
                else:
                    html2_lines.append("")

            elif line.startswith("+ "):
                # Added to text2
                html1_lines.append("")
                html2_lines.append(
                    f'<span style="background-color: #ccffcc;">{line[2:]}</span>'
                )

            elif line.startswith("  "):
                # Unchanged
                html1_lines.append(line[2:])
                html2_lines.append(line[2:])

            elif line.startswith("? "):
                # Change indicator - skip
                i += 1
                continue

            i += 1

        return ("<br>".join(html1_lines), "<br>".join(html2_lines))

    def find_common_sections(self, text1: str, text2: str) -> List[Dict]:
        """Find common sections between two documents."""
        paragraphs1 = [p.strip() for p in text1.split("\n\n") if p.strip()]
        paragraphs2 = [p.strip() for p in text2.split("\n\n") if p.strip()]

        common = []
        seen_p2 = set()  # Track matched paragraphs from text2 to avoid duplicates

        for i, p1 in enumerate(paragraphs1):
            # Quick pre-filter: skip if paragraph is too short
            if len(p1) < 10:
                continue
            for j, p2 in enumerate(paragraphs2):
                if j in seen_p2:
                    continue
                similarity = self.calculate_similarity(p1, p2)
                if similarity > 80:
                    common.append(
                        {
                            "paragraph_a_index": i,
                            "paragraph_b_index": j,
                            "similarity": similarity,
                            "content_a": p1[:100] + "..." if len(p1) > 100 else p1,
                            "content_b": p2[:100] + "..." if len(p2) > 100 else p2,
                        }
                    )
                    seen_p2.add(j)

        return sorted(common, key=lambda x: x["similarity"], reverse=True)
