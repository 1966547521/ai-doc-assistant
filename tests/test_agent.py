"""Tests for AI Agent tools and agent session."""
from unittest.mock import Mock, patch, MagicMock

from src.agent_tools import (
    ask_document, summarize_document,
    extract_info, translate_text, compare_documents,
    ALL_TOOLS,
)
from src.agent import AgentSession, stream_agent, TOOL_CALL_START, TOOL_CALL_END


def _make_mock_llm(response_content=""):
    """Create a mock LLM that supports stream() returning content as chunks."""
    mock = MagicMock()
    mock.invoke.return_value = Mock(content=response_content)

    def _mock_stream(_messages):
        """Simulate streaming: yield each character as a chunk."""
        for char in response_content:
            yield Mock(content=char)

    mock.stream = _mock_stream
    return mock


def _make_tool_call_text(tool_name, **kwargs):
    """Generate mock tool call text."""
    import json
    args = json.dumps(kwargs)
    return f"{TOOL_CALL_START}\n{{\"name\": \"{tool_name}\", \"arguments\": {args}}}\n{TOOL_CALL_END}"


class TestAgentTools:
    """Test each tool independently with mocked engines."""

    def test_all_tools_registered(self):
        assert len(ALL_TOOLS) == 7
        tool_names = {t.name for t in ALL_TOOLS}
        expected = {
            "ask_document", "summarize_document", "analyze_structure",
            "extract_info", "translate_text", "generate_report", "compare_documents"
        }
        assert tool_names == expected

    def test_ask_document_no_document(self, monkeypatch):
        monkeypatch.setattr("src.agent_tools._get_doc_text", lambda: "")
        result = ask_document.invoke({"question": "test"})
        assert "请先" in result

    def test_ask_document_no_rag_chain(self, monkeypatch):
        import streamlit as st
        monkeypatch.setattr("src.agent_tools._get_doc_text", lambda: "doc text")
        mock_engine = Mock()
        mock_engine.rag_chain = None
        with patch.dict(st.session_state, {"qa_engine": mock_engine}, clear=True):
            result = ask_document.invoke({"question": "test"})
            assert "尚未建立" in result

    def test_summarize_no_document(self, monkeypatch):
        monkeypatch.setattr("src.agent_tools._get_doc_text", lambda: "")
        result = summarize_document.invoke({"style": "short"})
        assert "请先" in result

    def test_extract_info_no_document(self, monkeypatch):
        monkeypatch.setattr("src.agent_tools._get_doc_text", lambda: "")
        result = extract_info.invoke({"target": "keywords"})
        assert "请先" in result

    def test_translate_no_document(self, monkeypatch):
        monkeypatch.setattr("src.agent_tools._get_doc_text", lambda: "")
        result = translate_text.invoke({"text": "", "target_language": "English"})
        assert "请提供" in result

    def test_compare_no_documents(self, monkeypatch):
        monkeypatch.setattr("src.agent_tools._get_doc_text", lambda: "")
        monkeypatch.setattr("src.agent_tools._get_compare_text", lambda: "")
        result = compare_documents.invoke({})
        assert "需要两篇" in result


class TestAgentSession:
    """Test AgentSession creation and ReAct loop."""

    def test_session_creation(self):
        mock_llm = _make_mock_llm()
        session = AgentSession(tools=ALL_TOOLS, llm=mock_llm)
        assert session.tool_map is not None
        assert len(session.tool_map) == 7

    def test_session_build_messages(self):
        mock_llm = _make_mock_llm()
        session = AgentSession(tools=ALL_TOOLS, llm=mock_llm)
        msgs = session._build_messages("hello")
        assert len(msgs) == 2
        assert msgs[0].type == "system"
        assert msgs[1].type == "human"

    def test_session_execute_tool_not_found(self):
        mock_llm = _make_mock_llm()
        session = AgentSession(tools=ALL_TOOLS, llm=mock_llm)
        result = session._execute_tool("nonexistent", {})
        assert "未找到" in result

    def test_session_no_tool_calls(self):
        """Agent responds with plain text (no tool calls)."""
        mock_llm = _make_mock_llm(response_content="你好，我是AI助手。")
        session = AgentSession(tools=ALL_TOOLS, llm=mock_llm)
        events = list(session.stream("hello"))
        event_types = {e["type"] for e in events}
        assert "token" in event_types
        assert "done" in event_types
        tokens = [e["content"] for e in events if e["type"] == "token"]
        assert "".join(tokens) == "你好，我是AI助手。"

    def test_session_with_text_tool_calls(self):
        """Agent correctly parses text-based tool calls."""
        tool_call_text = _make_tool_call_text("summarize_document", style="short")
        mock_llm = _make_mock_llm(response_content=tool_call_text)

        import streamlit as st
        with patch.dict(st.session_state, {
            "current_document_text": "test content",
            "summary_engine": Mock(generate_summary=Mock(return_value="文档摘要内容")),
        }, clear=True):
            session = AgentSession(tools=ALL_TOOLS, llm=mock_llm)
            events = list(session.stream("总结文档"))

        event_types = {e["type"] for e in events}
        assert "tool_start" in event_types
        assert "tool_end" in event_types

    def test_stream_with_error(self):
        mock_llm = _make_mock_llm()

        def _error_stream(_messages):
            raise RuntimeError("API error")
            yield  # Unreachable, makes it a generator

        mock_llm.stream = _error_stream
        session = AgentSession(tools=ALL_TOOLS, llm=mock_llm)
        events = list(session.stream("hello"))
        event_types = {e["type"] for e in events}
        assert "error" in event_types

    def test_stream_agent_convenience(self):
        mock_llm = _make_mock_llm(response_content="Hello")
        with patch("src.agent._build_llm_for_agent", return_value=mock_llm):
            events = list(stream_agent("hi", []))
            assert len(events) >= 1
