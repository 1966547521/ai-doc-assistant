"""Q&A tab for AI Document Assistant."""
import streamlit as st


def render_qa_tab():
    """Render the intelligent Q&A tab."""
    st.header("💬 智能问答")

    if not st.session_state.documents_uploaded:
        st.info("👈 请先在左侧侧边栏上传文档，然后即可基于文档内容进行智能问答。")
        st.markdown("""
        上传文档后您可以：
        - 对文档内容自由提问
        - 获得带参考来源的精准回答
        - 支持多轮对话记忆
        - 语义缓存加速重复问题
        """)
        return

    # ── Chat input at top (always visible, no scrolling needed) ──
    question = st.chat_input("基于文档内容，输入您的问题...")

    # ── Display chat history ──
    messages = st.session_state.memory_manager.get_messages()

    if messages:
        st.markdown('<div class="qa-messages-wrapper" id="qa-messages">', unsafe_allow_html=True)
    for msg in messages:
        avatar = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(avatar):
            st.write(msg["content"])
    if messages:
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Handle question ──
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            history = st.session_state.memory_manager.get_history()
            stream = st.session_state.qa_engine.stream_answer(question, history)
            full_answer = st.write_stream(stream)

            if full_answer:
                sources = st.session_state.qa_engine.get_sources(question, history)
                if sources:
                    with st.expander("📖 查看参考来源"):
                        for idx, source in enumerate(sources, 1):
                            cleaned = source.replace('\n', ' ').strip()[:150] + "..."
                            st.write(f"{idx}. {cleaned}")

        st.session_state.memory_manager.add_message("user", question)
        st.session_state.memory_manager.add_message("assistant", full_answer)

        if st.session_state.current_session_id:
            st.session_state.session_manager.add_message(st.session_state.current_session_id, "user", question)
            st.session_state.session_manager.add_message(st.session_state.current_session_id, "assistant", full_answer)

        st.rerun()
