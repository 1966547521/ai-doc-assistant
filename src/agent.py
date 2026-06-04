"""AI Agent — custom ReAct loop with text-based tool calling and session isolation.

Each conversation session gets its own AgentSession instance.
Uses text-based tool calling (XML-like tags) instead of native function calling
to avoid compatibility issues with DeepSeek reasoning models.
"""
import json
import re
from typing import Iterator, Dict, Any, List, Optional

from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, BaseMessage
)
from langchain_core.tools import BaseTool

from src.utils import get_llm
from src.logger import get_logger

logger = get_logger("agent")

# ── Tool calling format markers ───────────────────────────────
TOOL_CALL_START = "⚙️TOOL:"
TOOL_CALL_END = "⚙️END"

TOOL_PROMPT_INSTRUCTION = f"""
## 工具调用格式

当你确定需要使用工具时，先写出调用指令再停止。格式如下（纯文本，禁止用 markdown 代码块包裹）：

{TOOL_CALL_START}
{{"name": "summarize_document", "arguments": {{"style": "short"}}}}
{TOOL_CALL_END}

调用后立即停止输出，等待工具返回结果。收到结果后继续回复用户。
"""

SYSTEM_PROMPT = f"""你是 AI 智能文档助手，专门分析用户上传的文档。你必须通过工具来操作文档，不能凭空编造文档内容。

## 可用工具

| 工具 | 触发条件 | 参数 |
|------|---------|------|
| `ask_document` | 用户询问文档中的具体信息、数据、观点 | question: 问题文本 |
| `summarize_document` | 用户要求总结/概述/摘要/讲了什么 | style: short/detailed/bullet/executive/qa |
| `analyze_structure` | 用户问结构/大纲/章节/层级/目录 | 无参数 |
| `extract_info` | 用户要求提取关键词/行动项/主题 | target: keywords/actions/topics |
| `translate_text` | 用户要求翻译（未指定文本则翻译全文） | text: 原文（可空）, target_language: English/中文/日本語等 |
| `generate_report` | 用户要求完整报告/综合分析 | format_type: markdown/text, template: simple/standard/detailed |
| `compare_documents` | 用户要求对比两篇文档 | 无参数（需先在对比页面上传第二篇文档） |

{TOOL_PROMPT_INSTRUCTION}

## 决策规则

- 问文档内容 → `ask_document`；问概况 → `summarize_document`；问结构 → `analyze_structure`
- 一个任务需要多步时依次调用，如"总结后翻译"→先 summarize 后 translate
- 工具返回"请先上传文档"时，直接告知用户需先上传，不要反复重试
- 不需要工具的场景（打招呼、闲聊、问你能力范围）：直接简短回复
- 始终用中文回复，控制 3-5 句，除非用户要求详细

## 示例

用户："这篇文章的核心观点是什么？"
→ 调用 ask_document，question="核心观点是什么"

用户："帮我总结一下"
→ 调用 summarize_document，style="short"

用户："分析结构然后翻译成英文"
→ 先调 analyze_structure，收到结果后再调 translate_text，target_language="English"
"""

TOOL_NAMES_CN = {
    "ask_document": "文档问答",
    "summarize_document": "文档摘要",
    "analyze_structure": "结构分析",
    "extract_info": "信息提取",
    "translate_text": "文本翻译",
    "generate_report": "报告生成",
    "compare_documents": "文档对比",
}


def _build_llm_for_agent():
    """Create an LLM instance for the agent."""
    return get_llm(temperature=0.0)


def _parse_tool_calls(text: str) -> List[Dict]:
    """Parse text-based tool calls from LLM output.

    Expected format:
        ⚙️TOOL:
        {"name": "tool_name", "arguments": {...}}
        ⚙️END
    """
    pattern = rf'{re.escape(TOOL_CALL_START)}\s*(.*?)\s*{re.escape(TOOL_CALL_END)}'
    matches = re.findall(pattern, text, re.DOTALL)
    results = []
    for match in matches:
        try:
            data = json.loads(match.strip())
            if "name" in data:
                results.append({
                    "name": data["name"],
                    "args": data.get("arguments", data.get("args", {})),
                })
        except json.JSONDecodeError:
            logger.warning("Failed to parse tool call JSON: %s", match[:200])
    return results


def _strip_tool_calls(text: str) -> str:
    """Remove tool call markers from text, keeping only the user-visible content."""
    pattern = rf'{re.escape(TOOL_CALL_START)}.*?{re.escape(TOOL_CALL_END)}'
    return re.sub(pattern, '', text, flags=re.DOTALL).strip()


class AgentSession:
    """Per-session AI Agent with custom ReAct loop.

    Each session gets its own AgentSession instance with independent
    conversation state. Uses text-based tool calling for maximum LLM compatibility.

    Usage:
        session = AgentSession(tools=ALL_TOOLS)
        for event in session.stream("帮我总结文档", chat_history):
            handle(event)
    """

    MAX_ITERATIONS = 8

    def __init__(
        self,
        tools: List[BaseTool],
        llm=None,
        system_prompt: str = "",
    ):
        self.tools = tools
        self.tool_map = {t.name: t for t in tools}
        self.llm = llm or _build_llm_for_agent()
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    def _build_messages(
        self, query: str, chat_history: Optional[List[Dict]] = None
    ) -> List[BaseMessage]:
        """Build initial message list for this turn."""
        messages: List[BaseMessage] = [SystemMessage(content=self.system_prompt)]

        if chat_history:
            for msg in chat_history[-12:]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    # Clean tool call markers from history for display
                    clean = _strip_tool_calls(content)
                    messages.append(AIMessage(content=clean or content))

        messages.append(HumanMessage(content=query))
        return messages

    def _execute_tool(self, tool_name: str, tool_args: Dict) -> str:
        """Execute a tool and return its result."""
        tool = self.tool_map.get(tool_name)
        if not tool:
            return f"工具 '{tool_name}' 未找到。可用工具: {', '.join(self.tool_map.keys())}"

        try:
            logger.debug("Executing tool: %s(%s)", tool_name, str(tool_args)[:200])
            result = tool.invoke(tool_args)
            return str(result)
        except Exception as e:
            logger.error("Tool %s failed: %s", tool_name, str(e))
            return f"工具执行出错: {str(e)}"

    def _stream_llm(self, messages: List[BaseMessage]) -> Iterator[Dict[str, Any]]:
        """Stream LLM tokens with sliding-window tool call detection.

        Accumulates incoming tokens and scans for the tool marker (⚙️TOOL:)
        anywhere in the text, not just at the start. If found, stops streaming
        and silently collects the rest. Otherwise, outputs tokens in real-time
        in ~20-char batches for smooth rendering.

        Last event: {"type": "_stream_done", "content": "<full text>"}
        """
        all_chars = []
        streaming = False
        streamed_count = 0
        BATCH_SIZE = 20
        CHECK_BUFFER = len(TOOL_CALL_START) + 10
        HOLD_BACK = len(TOOL_CALL_START) - 1  # chars to hold back (might start tool marker)

        for chunk in self.llm.stream(messages):
            token = getattr(chunk, 'content', '') or ''
            if not token:
                continue
            all_chars.extend(token)
            text = "".join(all_chars)
            marker_at = text.find(TOOL_CALL_START)

            if marker_at >= 0:
                before_marker = text[streamed_count:marker_at]
                if before_marker:
                    for i in range(0, len(before_marker), BATCH_SIZE):
                        yield {"type": "token", "content": before_marker[i:i+BATCH_SIZE]}
                streaming = False
                streamed_count = marker_at
                continue

            if not streaming:
                if len(all_chars) >= CHECK_BUFFER:
                    streaming = True
                    full = "".join(all_chars)
                    for i in range(0, len(full), BATCH_SIZE):
                        yield {"type": "token", "content": full[i:i+BATCH_SIZE]}
                    streamed_count = len(full)
                continue

            new_text = text[streamed_count:]
            if len(new_text) > HOLD_BACK:
                safe = new_text[:-HOLD_BACK]
                for i in range(0, len(safe), BATCH_SIZE):
                    yield {"type": "token", "content": safe[i:i+BATCH_SIZE]}
                streamed_count += len(safe)

        content = "".join(all_chars)
        if not streaming and TOOL_CALL_START not in content and content.strip():
            for i in range(0, len(content), BATCH_SIZE):
                yield {"type": "token", "content": content[i:i+BATCH_SIZE]}
        yield {"type": "_stream_done", "content": content}

    def stream(
        self, query: str, chat_history: Optional[List[Dict]] = None
    ) -> Iterator[Dict[str, Any]]:
        """Run the ReAct loop with true LLM streaming output.

        Events yielded:
            {"type": "tool_start", "tool": "name", "display": "显示名"}
            {"type": "tool_end", "tool": "name"}
            {"type": "token", "content": "..."}    (real-time streaming)
            {"type": "done"}
            {"type": "error", "content": "..."}
        """
        messages = self._build_messages(query, chat_history)
        logger.debug("Agent stream: %d messages, query=%s", len(messages), query[:50])

        for iteration in range(self.MAX_ITERATIONS):
            content = ""
            try:
                for evt in self._stream_llm(messages):
                    if evt["type"] == "token":
                        yield evt
                    elif evt["type"] == "_stream_done":
                        content = evt.get("content", "")
            except Exception as e:
                logger.error("LLM stream failed (iter %d): %s", iteration, str(e))
                yield {"type": "error", "content": str(e)}
                return
            if not content:
                yield {"type": "done"}
                return

            # Check for text-based tool calls in the full response
            tool_calls = _parse_tool_calls(content)

            if tool_calls:
                # Strip tool call markers before adding to history
                clean_text = _strip_tool_calls(content)
                if clean_text:
                    messages.append(AIMessage(content=clean_text))

                for tc in tool_calls:
                    tool_name = tc["name"]
                    display = TOOL_NAMES_CN.get(tool_name, tool_name)

                    # Build args preview for UI display
                    args_parts = []
                    for k, v in tc.get("args", {}).items():
                        if v and v != "" and k != "text":  # skip full text
                            v_str = str(v)
                            if len(v_str) > 20:
                                v_str = v_str[:20] + "..."
                            args_parts.append(f"{k}={v_str}")
                    args_preview = ", ".join(args_parts)

                    yield {
                        "type": "tool_start",
                        "tool": tool_name,
                        "display": display,
                        "args_preview": args_preview,
                    }

                    result = self._execute_tool(tool_name, tc["args"])

                    messages.append(HumanMessage(
                        content=f"工具 {tool_name} 的执行结果:\n{result}\n\n请基于此结果继续回复用户。"
                    ))

                    yield {"type": "tool_end", "tool": tool_name}
                    yield {"type": "reasoning", "content": "正在分析工具结果..."}

                # Loop back to LLM with tool results
                continue

            # --- Final response (no tool calls) ---
            # Tokens were already streamed by _stream_llm
            break

        else:
            yield {
                "type": "error",
                "content": "Agent 执行步骤过多，已中止。请简化你的请求。",
            }
            return

        yield {"type": "done"}


# ── Factory for session-based agents ──────────────────────────

def get_or_create_session_agent(
    session_id: Optional[str] = None,
    tools: Optional[List] = None,
) -> AgentSession:
    """Get or create an AgentSession for a given session ID."""
    if tools is None:
        from src.agent_tools import ALL_TOOLS
        tools = ALL_TOOLS

    if session_id is None:
        return AgentSession(tools=tools)

    import streamlit as st
    cache_key = f"_agent_session_{session_id}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = AgentSession(tools=tools)
    return st.session_state[cache_key]


def stream_agent(
    query: str,
    chat_history: Optional[List[Dict]] = None,
    tools: Optional[List] = None,
    system_prompt: str = "",
    session_id: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    """Convenience wrapper for AgentSession.stream()."""
    agent = get_or_create_session_agent(session_id=session_id, tools=tools)
    return agent.stream(query, chat_history)
