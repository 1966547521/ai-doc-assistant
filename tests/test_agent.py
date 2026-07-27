"""Behavior tests for structured Agent routing and streaming events."""

from langchain_core.tools import StructuredTool

from src.agent import AgentRoute, AgentSession, ToolInvocation, stream_agent


class FakeStructuredLLM:
    """Deterministic boundary double for the external structured-output API."""

    def __init__(self, route=None, error=None):
        self.route = route
        self.error = error
        self.schema = None
        self.messages = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.messages = messages
        if self.error:
            raise self.error
        return self.route


def _tool(name, result, calls):
    if name == "ask_document":
        def run(question: str):
            """Answer a document question."""
            calls.append((name, {"question": question}))
            return result
    elif name == "summarize_document":
        def run(style: str = "short"):
            """Summarize a document."""
            calls.append((name, {"style": style}))
            return result
    elif name == "translate_text":
        def run(text: str, target_language: str):
            """Translate document text."""
            calls.append((name, {
                "text": text,
                "target_language": target_language,
            }))
            return result
    else:
        def run():
            """Run a document operation without arguments."""
            calls.append((name, {}))
            return result

    return StructuredTool.from_function(
        func=run,
        name=name,
        description=f"用于测试 {name} 的真实工具包装。",
    )


def _event_text(events):
    return "".join(event["content"] for event in events if event["type"] == "token")


class TestAgentSession:
    def test_session_builds_messages_without_marker_protocol_processing(self):
        llm = FakeStructuredLLM(
            AgentRoute(response="你好")
        )
        session = AgentSession(tools=[], llm=llm)

        messages = session._build_messages(
            "继续",
            [{"role": "assistant", "content": "保留这段 ⚙️TOOL: 普通文本"}],
        )

        assert messages[1].content == "保留这段 ⚙️TOOL: 普通文本"

    def test_structured_route_executes_tool_and_keeps_stream_event_contract(self):
        calls = []
        tool = _tool("summarize_document", "可靠摘要", calls)
        llm = FakeStructuredLLM(
            AgentRoute(
                tool_calls=[
                    ToolInvocation(
                        name="summarize_document",
                        arguments={"style": "short"},
                    )
                ]
            )
        )
        session = AgentSession(tools=[tool], llm=llm)

        events = list(session.stream("总结文档"))

        assert llm.schema is AgentRoute
        assert calls == [("summarize_document", {"style": "short"})]
        assert [event["type"] for event in events] == [
            "tool_start",
            "tool_end",
            "token",
            "done",
        ]
        assert _event_text(events) == "可靠摘要"

    def test_multiple_structured_tool_calls_preserve_order_and_results(self):
        calls = []
        tools = [
            _tool("analyze_structure", "结构结果", calls),
            _tool("translate_text", "翻译结果", calls),
        ]
        llm = FakeStructuredLLM(
            AgentRoute(
                tool_calls=[
                    ToolInvocation(name="analyze_structure"),
                    ToolInvocation(
                        name="translate_text",
                        arguments={"text": "", "target_language": "English"},
                    ),
                ]
            )
        )

        events = list(AgentSession(tools=tools, llm=llm).stream("分析后翻译"))

        assert [name for name, _ in calls] == ["analyze_structure", "translate_text"]
        assert _event_text(events) == "结构结果\n\n翻译结果"

    def test_direct_response_is_emitted_as_stream_tokens(self):
        llm = FakeStructuredLLM(AgentRoute(response="你好，我可以分析文档。"))

        events = list(AgentSession(tools=[], llm=llm).stream("你好"))

        assert _event_text(events) == "你好，我可以分析文档。"
        assert events[-1] == {"type": "done"}

    def test_document_qa_uses_streaming_handler_without_waiting_for_full_tool_result(self):
        calls = []
        tool = _tool("ask_document", "不应调用同步工具", calls)
        llm = FakeStructuredLLM(
            AgentRoute(
                tool_calls=[ToolInvocation(name="ask_document", arguments={"question": "预算？"})]
            )
        )

        def stream_document_answer(arguments):
            assert arguments == {"question": "预算？"}
            yield "第一段"
            yield "第二段"

        events = list(AgentSession(
            tools=[tool],
            llm=llm,
            streaming_handlers={"ask_document": stream_document_answer},
        ).stream("预算？"))

        assert calls == []
        assert _event_text(events) == "第一段第二段"
        assert [event["type"] for event in events] == [
            "tool_start", "token", "token", "tool_end", "done",
        ]

    def test_route_failure_falls_back_to_document_qa_tool(self):
        calls = []
        ask_tool = _tool("ask_document", "来自真实检索链的回答", calls)
        llm = FakeStructuredLLM(error=ValueError("invalid structured output"))

        events = list(AgentSession(tools=[ask_tool], llm=llm).stream("预算是多少？"))

        assert calls == [("ask_document", {"question": "预算是多少？"})]
        assert _event_text(events) == "来自真实检索链的回答"
        assert any(event["type"] == "reasoning" for event in events)
        assert not any(event["type"] == "error" for event in events)

    def test_route_failure_without_qa_tool_reports_clear_error(self):
        llm = FakeStructuredLLM(error=RuntimeError("provider rejected schema"))

        events = list(AgentSession(tools=[], llm=llm).stream("你好"))

        assert events == [{
            "type": "error",
            "content": "意图解析失败，请检查模型是否支持结构化输出及 API 配置。",
        }]

    def test_unknown_tool_is_reported_without_fabricated_result(self):
        llm = FakeStructuredLLM(
            AgentRoute(
                tool_calls=[ToolInvocation(name="summarize_document")]
            )
        )

        events = list(AgentSession(tools=[], llm=llm).stream("总结"))

        assert events[0]["type"] == "error"
        assert "summarize_document" in events[0]["content"]
        assert not any(event["type"] == "token" for event in events)


def test_stream_agent_convenience_uses_session_agent(monkeypatch):
    llm = FakeStructuredLLM(AgentRoute(response="Hello"))
    monkeypatch.setattr("src.agent._build_llm_for_agent", lambda: llm)

    events = list(stream_agent("hi", []))

    assert _event_text(events) == "Hello"
