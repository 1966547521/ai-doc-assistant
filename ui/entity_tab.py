"""Entity extraction tab for AI Document Assistant."""
import streamlit as st
from ui.utils import persist_content


def render_entity_tab():
    """Render the entity extraction tab."""
    st.header("🔍 实体提取")

    if not st.session_state.documents_uploaded:
        st.info("👈 请先在左侧侧边栏上传文档，然后即可提取关键词、行动项和主题。")
        st.markdown("""
        提取类型：
        - **关键词**：自动识别文档中的核心关键词和术语
        - **行动项**：提取文档中的待办任务和行动要点
        - **主题**：识别文档讨论的主要话题
        """)
        return

    if "extraction_results" not in st.session_state:
        st.session_state.extraction_results = {}

    col1, col2 = st.columns(2)

    with col1:
        if st.button("提取关键词"):
            with st.spinner("提取中..."):
                extractor = st.session_state.keyword_extractor
                text = st.session_state.current_document_text
                stream = extractor.stream_extract_key_terms(text)
                st.session_state.extraction_results["关键词"] = "".join(stream)
                _save_extraction()

    with col2:
        if st.button("提取行动项"):
            with st.spinner("提取中..."):
                extractor = st.session_state.keyword_extractor
                text = st.session_state.current_document_text
                stream = extractor.stream_extract_actions(text)
                st.session_state.extraction_results["行动项"] = "".join(stream)
                _save_extraction()

    if st.button("提取主题"):
        with st.spinner("提取中..."):
            extractor = st.session_state.keyword_extractor
            text = st.session_state.current_document_text
            stream = extractor.stream_extract_topics(text)
            st.session_state.extraction_results["主题"] = "".join(stream)
            _save_extraction()

    if st.session_state.extraction_results:
        st.divider()
        with st.expander("📄 上次提取结果", expanded=False):
            for label, content in st.session_state.extraction_results.items():
                st.subheader(label)
                st.write(content)


def _save_extraction():
    """Persist extraction results only when new data is extracted."""
    if st.session_state.extraction_results:
        persist_content(
            st.session_state.get("current_session_id"),
            "extraction_results",
            dict(st.session_state.extraction_results),
        )
