"""Structured intent router for the Streamlit document assistant.

The model returns a validated Pydantic route instead of embedding tool calls
inside user-visible text.  Tool execution remains session-isolated and emits
the same event protocol consumed by ``ui.agent_chat``.
"""

from typing import Any, Dict, Iterator, List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

from src.logger import get_logger

logger = get_logger("agent")

ToolName = Literal[
    "ask_document",
    "summarize_document",
    "analyze_structure",
    "extract_info",
    "translate_text",
    "generate_report",
    "compare_documents",
]


class ToolInvocation(BaseModel):
    """One validated tool operation selected by the intent router."""

    name: ToolName = Field(description="要执行的文档工具名称")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="与工具参数定义匹配的参数；没有参数时返回空对象",
    )


class AgentRoute(BaseModel):
    """Structured decision for one user turn.

    Exactly one of ``response`` and ``tool_calls`` must be populated.  A
    response is suitable only for greetings, casual conversation, or capability
    questions; document-dependent requests must use tools.
    """

    response: Optional[str] = Field(
        default=None,
        description="无需文档工具时直接给用户的中文答复，否则为 null",
    )
    tool_calls: List[ToolInvocation] = Field(
        default_factory=list,
        description="按执行顺序排列的一个或多个工具调用",
    )

    @model_validator(mode="after")
    def validate_single_route(self):
        has_response = bool(self.response and self.response.strip())
        has_tools = bool(self.tool_calls)
        if has_response == has_tools:
            raise ValueError("response 和 tool_calls 必须且只能提供一个")
        return self


SYSTEM_PROMPT = """你是 AI 智能文档助手的意图路由器。根据用户请求返回结构化决策，不要在普通文本中编写工具指令。

工具选择规则：
- ask_document：询问文档中的具体信息、数据、事实、观点或需要查证的内容。
- summarize_document：总结、概述、摘要；style 可用 short/detailed/bullet/executive/qa。
- analyze_structure：分析结构、大纲、章节、层级或目录。
- extract_info：提取关键词、行动项或主题；target 可用 keywords/actions/topics。
- translate_text：翻译指定文本或全文；提供 text 与 target_language。
- generate_report：生成综合报告；format_type 可用 markdown/text，template 可用 simple/standard/detailed。
- compare_documents：比较当前文档和对比文档。

文档相关请求必须选择工具，不能凭空回答。复合请求可以按顺序返回多个工具调用。只有打招呼、闲聊或询问能力范围时才填写 response。始终使用中文。
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
    """Create the real configured LLM lazily."""
    from src.utils import get_llm

    return get_llm(temperature=0.0)


class AgentSession:
    """Per-session structured router with the Streamlit event contract."""

    TOKEN_BATCH_SIZE = 20

    def __init__(
        self,
        tools: List[BaseTool],
        llm=None,
        system_prompt: str = "",
        streaming_handlers: Optional[Dict[str, Any]] = None,
    ):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        self.llm = llm or _build_llm_for_agent()
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.streaming_handlers = streaming_handlers or {}

    def _build_messages(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None,
    ) -> List[BaseMessage]:
        """Build router context without interpreting assistant text as commands."""
        messages: List[BaseMessage] = [SystemMessage(content=self.system_prompt)]
        for message in (chat_history or [])[-12:]:
            role = message.get("role", "")
            content = message.get("content", "")
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=query))
        return messages

    def _route(self, messages: List[BaseMessage]) -> AgentRoute:
        """Ask the configured LLM for a schema-validated routing decision."""
        if not hasattr(self.llm, "with_structured_output"):
            raise TypeError("当前模型客户端不支持结构化输出")
        router = self.llm.with_structured_output(AgentRoute)
        route = router.invoke(messages)
        if isinstance(route, AgentRoute):
            return route
        return AgentRoute.model_validate(route)

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute one registered tool and turn failures into visible results."""
        tool = self.tool_map.get(tool_name)
        if tool is None:
            raise LookupError(f"工具 '{tool_name}' 未注册")
        try:
            logger.debug("Executing tool: %s(%s)", tool_name, str(tool_args)[:200])
            return str(tool.invoke(tool_args))
        except Exception as exc:
            logger.error("Tool %s failed: %s", tool_name, exc)
            return f"❌ 工具执行出错: {exc}"

    @staticmethod
    def _args_preview(arguments: Dict[str, Any]) -> str:
        parts = []
        for key, value in arguments.items():
            if value in (None, "") or key == "text":
                continue
            text = str(value)
            if len(text) > 20:
                text = f"{text[:20]}..."
            parts.append(f"{key}={text}")
        return ", ".join(parts)

    def _token_events(self, content: str) -> Iterator[Dict[str, Any]]:
        for offset in range(0, len(content), self.TOKEN_BATCH_SIZE):
            yield {
                "type": "token",
                "content": content[offset:offset + self.TOKEN_BATCH_SIZE],
            }

    def _tool_events(
        self,
        tool_calls: List[ToolInvocation],
    ) -> Iterator[Dict[str, Any]]:
        results = []
        for call in tool_calls:
            if call.name not in self.tool_map:
                yield {
                    "type": "error",
                    "content": f"意图路由选择了未注册工具: {call.name}",
                }
                return

            yield {
                "type": "tool_start",
                "tool": call.name,
                "display": TOOL_NAMES_CN.get(call.name, call.name),
                "args_preview": self._args_preview(call.arguments),
            }
            streaming_handler = self.streaming_handlers.get(call.name)
            if streaming_handler:
                try:
                    for chunk in streaming_handler(call.arguments):
                        if chunk:
                            yield {"type": "token", "content": str(chunk)}
                except Exception as exc:
                    logger.error("Streaming tool %s failed: %s", call.name, exc)
                    yield {
                        "type": "error",
                        "content": f"❌ 工具执行出错: {exc}",
                    }
                    return
            else:
                results.append(self._execute_tool(call.name, call.arguments))
            yield {"type": "tool_end", "tool": call.name}

        if results:
            combined = "\n\n".join(results)
            yield from self._token_events(combined)
        yield {"type": "done"}

    def stream(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Route one turn and yield UI-compatible streaming events.

        Events remain ``tool_start``, ``tool_end``, ``reasoning``, ``token``,
        ``done`` and ``error``. If structured routing fails, an available
        ``ask_document`` tool receives the original question, preserving a real
        RAG path without inventing a response.
        """
        messages = self._build_messages(query, chat_history)
        try:
            route = self._route(messages)
        except Exception as exc:
            logger.error("Structured intent routing failed: %s", exc)
            if "ask_document" not in self.tool_map:
                yield {
                    "type": "error",
                    "content": "意图解析失败，请检查模型是否支持结构化输出及 API 配置。",
                }
                return
            yield {
                "type": "reasoning",
                "content": "结构化意图解析失败，正在转为文档问答。",
            }
            fallback = ToolInvocation(
                name="ask_document",
                arguments={"question": query},
            )
            yield from self._tool_events([fallback])
            return

        if route.tool_calls:
            yield from self._tool_events(route.tool_calls)
            return

        yield from self._token_events(route.response or "")
        yield {"type": "done"}


def get_or_create_session_agent(
    session_id: Optional[str] = None,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: str = "",
    streaming_handlers: Optional[Dict[str, Any]] = None,
) -> AgentSession:
    """Get or create an isolated AgentSession for a Streamlit session."""
    if tools is None:
        from src.agent_tools import ALL_TOOLS, STREAMING_TOOL_HANDLERS

        tools = ALL_TOOLS
        streaming_handlers = streaming_handlers or STREAMING_TOOL_HANDLERS

    if session_id is None:
        return AgentSession(
            tools=tools,
            system_prompt=system_prompt,
            streaming_handlers=streaming_handlers,
        )

    import streamlit as st

    cache_key = f"_agent_session_{session_id}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = AgentSession(
            tools=tools,
            system_prompt=system_prompt,
            streaming_handlers=streaming_handlers,
        )
    return st.session_state[cache_key]


def stream_agent(
    query: str,
    chat_history: Optional[List[Dict]] = None,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: str = "",
    session_id: Optional[str] = None,
    streaming_handlers: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    """Convenience wrapper used by the Streamlit chat page."""
    agent = get_or_create_session_agent(
        session_id=session_id,
        tools=tools,
        system_prompt=system_prompt,
        streaming_handlers=streaming_handlers,
    )
    return agent.stream(query, chat_history)
