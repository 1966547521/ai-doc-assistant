"""Agent Chat UI — true streaming via st.write_stream.

st.write_stream sends each yielded string as a separate frontend delta,
achieving real character-by-character streaming in the browser.
Tool calls render in a separate status container.
"""
import time

import streamlit as st
from src.agent import stream_agent
from src.logger import get_logger

logger = get_logger("ui.agent_chat")

AGENT_CHAT_CSS = """
<style>
.tool-badge {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 0.8rem; font-weight: 500; margin: 2px 4px;
}
.tool-badge.active { background: #e8f0fe; color: #1967d2; }
.tool-badge.done  { background: #e6f4ea; color: #137333; }

/* Typing cursor animation */
.typing-cursor::after {
    content: "▌";
    animation: blink 1s step-end infinite;
    color: #4a6fa5;
}
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

/* Pulsing dot for "thinking" state */
.thinking-dots { display: inline-flex; gap: 4px; align-items: center; }
.thinking-dots span {
    width: 6px; height: 6px; border-radius: 50%; background: #4a6fa5;
    animation: pulse 1.4s ease-in-out infinite both;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes pulse { 0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); } 40% { opacity: 1; transform: scale(1); } }
</style>
"""


def _messages_key() -> str:
    sid = st.session_state.get("current_session_id", "")
    return f"_agent_messages_{sid}" if sid else "_agent_messages_default"


def _restore_messages():
    sid = st.session_state.get("current_session_id")
    if not sid: return
    mk = _messages_key()
    if mk in st.session_state and st.session_state[mk]: return
    mgr = st.session_state.get("session_manager")
    if not mgr: return
    s = mgr.get_session_by_id(sid)
    if not s: return
    h = getattr(s, 'chat_history', []) or []
    if h:
        st.session_state[mk] = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in h if m.get("content")
        ]


def render_agent_chat():
    st.markdown(AGENT_CHAT_CSS, unsafe_allow_html=True)
    st.header("🧠 AI 助手")

    if not st.session_state.get("documents_uploaded"):
        st.info("👈 请先在左侧侧边栏上传文档并点击「开始处理」，然后与我对话。")
        with st.expander("💡 上传文档后，你可以这样对我说", expanded=True):
            st.markdown("""
| 你可以说 | Agent 自动... |
|----------|-------------|
| "这篇文章讲了什么？" | 调用摘要 |
| "分析文档结构" | 调用结构分析 |
| "提取关键词" | 调用信息提取 |
| "翻译成英文" | 调用翻译 |
| "生成报告" | 调用报告生成 |
            """)
        return

    _restore_messages()
    msg_key = _messages_key()

    # ── Clear button ──
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🗑️ 清空", key=f"clr_{msg_key}"):
            st.session_state[msg_key] = []
            st.session_state.pop(
                f"_agent_session_{st.session_state.get('current_session_id', '')}", None
            )
            st.rerun()

    # ── Message list ──
    messages = st.session_state.setdefault(msg_key, [])

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        with st.chat_message(role):
            if tool_calls:
                with st.expander(f"🔧 {len(tool_calls)} 个工具", expanded=False):
                    for tc in tool_calls:
                        name = tc.get("display", tc.get("tool", "?"))
                        st.markdown(
                            f"<span class='tool-badge done'>{name}</span>",
                            unsafe_allow_html=True,
                        )
            if content:
                st.markdown(content)
            elif tool_calls:
                st.caption("_任务完成_")

    # ── Input ──
    if query := st.chat_input("用自然语言告诉我你想做什么..."):
        messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        chat_history = messages[:-1]
        sid = st.session_state.get("current_session_id")
        tool_events = []

        with st.chat_message("assistant"):
            tool_status = st.empty()

            def _token_gen():
                """Generator that yields text tokens for st.write_stream."""
                nonlocal tool_events
                tool_badges = []
                for event in stream_agent(query, chat_history, session_id=sid):
                    etype = event.get("type", "")

                    if etype == "tool_start":
                        name = event.get("tool", "?")
                        display = event.get("display", name)
                        tool_events.append({"tool": name, "display": display})
                        tool_badges.append(
                            f'<span class="tool-badge active">🔧 {display}</span>'
                        )
                        tool_status.markdown(
                            " ".join(tool_badges), unsafe_allow_html=True
                        )

                    elif etype == "tool_end":
                        tool_status.empty()
                        tool_badges = [b.replace("active", "done") for b in tool_badges]

                    elif etype == "token":
                        yield event.get("content", "")
                        time.sleep(0.03)

                    elif etype == "error":
                        yield f"\n\n❌ {event['content']}"

                    elif etype == "done":
                        if tool_badges:
                            tool_status.markdown(
                                " ".join(tool_badges), unsafe_allow_html=True
                            )

            thinking = st.empty()
            thinking.markdown(
                '<div class="thinking-dots"><span></span><span></span><span></span></div>',
                unsafe_allow_html=True,
            )

            full_response = st.write_stream(_token_gen())
            thinking.empty()

        messages.append({
            "role": "assistant",
            "content": full_response or "",
            "tool_calls": tool_events,
        })

        if sid:
            try:
                mgr = st.session_state.get("session_manager")
                if mgr:
                    mgr.add_message(sid, "user", query)
                    if full_response:
                        mgr.add_message(sid, "assistant", full_response)
            except Exception as e:
                logger.warning("Persist: %s", str(e))
