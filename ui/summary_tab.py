"""Summary tab for AI Document Assistant."""
import streamlit as st
from ui.utils import persist_content


def render_summary_tab():
    """Render the document summary tab."""
    st.header("📝 文档摘要")

    if not st.session_state.documents_uploaded:
        st.info("👈 请先在左侧侧边栏上传文档，然后即可生成多种格式的摘要。")
        return

    # Show previously generated summary
    saved = st.session_state.get("_summary_result")
    if saved:
        with st.expander(f"📄 上次摘要 · {saved.get('type', '')}", expanded=False):
            st.write(saved.get("content", ""))
        st.divider()

    summary_type = st.selectbox(
        "选择摘要类型",
        ["简明摘要", "详细摘要", "要点列表", "执行摘要", "问答式摘要"]
    )

    if st.button("生成摘要"):
        with st.spinner("正在生成摘要..."):
            engine = st.session_state.summary_engine
            text = st.session_state.current_document_text

            with st.empty():
                if summary_type == "简明摘要":
                    stream = engine.stream_summary(text, length="short")
                elif summary_type == "详细摘要":
                    stream = engine.stream_summary(text, length="detailed")
                elif summary_type == "要点列表":
                    stream = engine.stream_bullet_summary(text)
                elif summary_type == "执行摘要":
                    stream = engine.stream_executive_summary(text)
                else:
                    stream = engine.stream_summary_with_questions(text)

                summary = st.write_stream(stream)

            persist_content(
                st.session_state.get("current_session_id"),
                "_summary_result",
                {"type": summary_type, "content": summary},
            )
            st.session_state.cache_manager.cache_document_summary(text, summary)
            st.rerun()
