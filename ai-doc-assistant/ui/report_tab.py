"""Report generation tab for AI Document Assistant."""
import streamlit as st
from ui.utils import persist_content


def render_report_tab():
    """Render the report generation tab."""
    st.header("📊 报告生成")

    if not st.session_state.documents_uploaded:
        st.info("👈 请先在左侧侧边栏上传文档，然后即可一键生成综合分析报告。")
        st.markdown("""
        报告包含：
        - 文档基本信息（长度、章节数、深度）
        - 内容摘要
        - 关键词列表
        - 文档结构树
        - 支持 Markdown / 纯文本格式导出
        """)
        return

    # Show previously generated report
    saved = st.session_state.get("_report_result")
    if saved:
        st.subheader(f"上次报告 · {saved.get('format', '')}")
        with st.expander("查看报告", expanded=True):
            st.markdown(saved.get("content", ""))
        _report_download(saved["content"], saved["format"])
        st.divider()

    col1, col2 = st.columns(2)

    with col1:
        report_format = st.selectbox(
            "报告格式",
            options=["markdown", "text"],
            format_func=lambda x: "Markdown" if x == "markdown" else "纯文本",
            index=0
        )

    with col2:
        st.radio(
            "报告类型",
            options=["完整报告", "简洁报告"],
            horizontal=True
        )

    if st.button("生成报告"):
        generator = st.session_state.report_generator
        text = st.session_state.current_document_text

        st.subheader("生成的报告")

        with st.empty():
            if report_format == "markdown":
                stream = generator.stream_generate_markdown_report(text)
            else:
                stream = generator.stream_generate_full_report(text)

            report = st.write_stream(stream)

        st.session_state._report_result = {"content": report, "format": report_format}
        persist_content(
            st.session_state.get("current_session_id"),
            "_report_result",
            {"content": report, "format": report_format},
        )
        _report_download(report, report_format)


def _report_download(content: str, fmt: str):
    """Render download button for report."""
    ext = "md" if fmt == "markdown" else "txt"
    mime = "text/markdown" if fmt == "markdown" else "text/plain"
    st.download_button(
        label=f"下载报告 (.{ext})",
        data=content,
        file_name=f"document_report.{ext}",
        mime=mime,
    )
