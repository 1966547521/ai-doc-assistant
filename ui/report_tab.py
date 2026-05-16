"""Report generation tab for AI Document Assistant."""
import re

import streamlit as st
from ui.utils import persist_content

TEMPLATE_LABELS = {
    "simple": "简洁 — 仅统计+摘要",
    "standard": "标准 — 全部分析",
    "detailed": "详尽 — 标准 + AI深度润色",
}

REPORT_CSS = """
<style>
.report-stage { margin-bottom: 0.3em; font-weight: 500; color: #4a6fa5; }
.report-stage .stage-icon { margin-right: 4px; }
</style>
"""


def _extract_stats(text: str) -> dict:
    words = re.findall(r'\b\w+\b', text)
    sentences = [s.strip() for s in re.split(r'[。！？.!?]+', text) if s.strip()]
    return {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_sentence_len": round(len(words) / max(len(sentences), 1)),
        "reading_time": max(1, round(len(words) / 200)),
    }


def _render_stats_cards(stats: dict):
    """Render metrics in a horizontal row of cards."""
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📝 总字符", f"{stats['char_count']:,}")
    c2.metric("🔤 英文词", f"{stats['word_count']:,}")
    c3.metric("📏 句子数", f"{stats['sentence_count']:,}")
    c4.metric("⏱ 阅读", f"{stats['reading_time']} 分钟")
    c5.metric("📊 句长", f"{stats['avg_sentence_len']} 词")


def render_report_tab():
    """Render the report generation tab."""
    st.markdown(REPORT_CSS, unsafe_allow_html=True)
    st.header("📊 报告生成")

    if not st.session_state.documents_uploaded:
        st.info("👈 请先在左侧侧边栏上传文档，然后即可一键生成综合分析报告。")
        st.markdown("""
        | 模板 | 包含内容 |
        |------|---------|
        | 简洁 | 统计 + 摘要 |
        | 标准 | 统计 + 摘要 + 关键词 + 结构 + 洞察 |
        | 详尽 | 标准全部 + AI 深度润色（核心论点、盲区、建议） |
        """)
        return

    # ── Previous report (collapsed) ──
    saved = st.session_state.get("_report_result")
    if saved:
        with st.expander(
            f"📄 上次报告 · {saved.get('template_label', '')} · {saved.get('format', '')}",
            expanded=False,
        ):
            st.markdown(saved.get("content", ""))
        _report_download(saved["content"], saved["format"], key="dl_saved")
        st.divider()

    # ── Controls ──
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        template = st.selectbox(
            "报告模板",
            options=["simple", "standard", "detailed"],
            format_func=lambda x: TEMPLATE_LABELS[x],
            index=1,
            key="report_template",
        )
    with c2:
        report_format = st.selectbox(
            "导出格式",
            options=["markdown", "text"],
            format_func=lambda x: "Markdown" if x == "markdown" else "纯文本",
            index=0,
            key="report_format",
        )
    with c3:
        st.write("")
        st.write("")
        generate = st.button("🚀 生成报告", use_container_width=True)

    if not generate:
        return

    generator = st.session_state.report_generator
    text = st.session_state.current_document_text
    enhance = template == "detailed"

    # ── Stage 1: Stats ──
    st.markdown('<p class="report-stage"><span class="stage-icon">📊</span> 文档统计</p>', unsafe_allow_html=True)
    stats = _extract_stats(text)
    _render_stats_cards(stats)

    # ── Stage 2: Generate base report (always streams) ──
    st.markdown('<p class="report-stage"><span class="stage-icon">📄</span> 生成报告</p>', unsafe_allow_html=True)

    if report_format == "markdown":
        stream = generator.stream_generate_markdown_report(
            text, template="detailed" if enhance else template
        )
    else:
        stream = generator.stream_generate_full_report(text)

    report = st.write_stream(stream)

    # ── Stage 3: AI deep polish (streaming) ──
    if enhance:
        st.markdown(
            '<p class="report-stage"><span class="stage-icon">🧠</span> AI 深度润色中...</p>',
            unsafe_allow_html=True,
        )
        enhancer = generator._get_enhancer()
        report = st.write_stream(enhancer.stream_enhance_report(report, text))

    # ── Final report in bordered container ──
    with st.container(border=True):
        st.markdown(report)

    # ── Save & download ──
    st.session_state._report_result = {
        "content": report,
        "format": report_format,
        "template_label": TEMPLATE_LABELS[template],
    }
    persist_content(
        st.session_state.get("current_session_id"),
        "_report_result",
        {"content": report, "format": report_format},
    )
    _report_download(report, report_format, key="dl_new")


def _report_download(content: str, fmt: str, key: str = "dl_report"):
    """Render download button for report."""
    ext = "md" if fmt == "markdown" else "txt"
    mime = "text/markdown" if fmt == "markdown" else "text/plain"
    st.download_button(
        label=f"📥 下载报告 (.{ext})",
        data=content,
        file_name=f"document_report.{ext}",
        mime=mime,
        key=key,
    )
