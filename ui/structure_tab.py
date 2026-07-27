"""Structure analysis tab for AI Document Assistant."""
import streamlit as st
from ui.utils import render_outline


def render_structure_tab():
    """Render the document structure analysis tab."""
    st.header("🏗️ 文档结构分析")

    if not st.session_state.documents_uploaded:
        st.info("👈 请先在左侧侧边栏上传文档，然后即可自动分析文档结构。")
        st.markdown("""
        AI 会从两个层面分析：
        - **表层结构**：标题层级、章节统计、目录生成
        - **深层理解**：文档类型识别、各章节内容总结、组织逻辑概述、质量评估
        """)
        return

    results = st.session_state.analysis_results

    # ── No structure ──────────────────────────────────────────
    if not results.get("has_structure", True):
        st.warning(results.get("message", "本文无明显结构"))
        if results.get("preamble"):
            st.subheader("📝 文档内容")
            st.write(results["preamble"][:2000])
        return

    # ── LLM Insights (deep analysis) ──────────────────────────
    doc_type = results.get("doc_type", "")
    doc_purpose = results.get("doc_purpose", "")
    overview = results.get("overview", "")

    if doc_type or overview:
        st.subheader("🧠 AI 结构理解")

        if doc_type:
            cols = st.columns([1, 3])
            with cols[0]:
                st.metric("文档类型", doc_type)
            with cols[1]:
                if doc_purpose:
                    st.caption(f"📌 {doc_purpose}")

        if overview:
            st.markdown(f"<div class='insight-card'>{overview}</div>", unsafe_allow_html=True)

        st.divider()

    # ── Stats row ─────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("标题数量", results.get("total_headings", 0))
    with col2:
        st.metric("章节数量", results.get("total_sections", 0))
    with col3:
        st.metric("文档深度", results.get("depth", 0))

    # ── Section summaries from LLM ────────────────────────────
    section_summaries = results.get("section_summaries", [])
    if section_summaries:
        st.divider()
        st.subheader("📋 章节内容速览")
        for item in section_summaries:
            title = item.get("title", "")
            summary = item.get("summary", "")
            if summary:
                st.markdown(f"**{title}**：{summary}")

    # ── Section stats + heading distribution ──────────────────
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 章节统计")
        stats = results.get("section_stats", {})
        if stats:
            st.write(f"平均长度: **{stats.get('avg_length', 0)}** 字符")
            st.write(f"最长章节: **{stats.get('max_length', 0)}** 字符")
            st.write(f"最短章节: **{stats.get('min_length', 0)}** 字符")
            st.write(f"总字符数: **{stats.get('total_chars', 0)}** 字符")

    with col2:
        st.subheader("📑 标题分布")
        headings = results.get("headings", [])
        dist = st.session_state.application_service.structure_analyzer.get_level_distribution(headings)
        st.write(f"一级 (h1): **{dist.get('h1', 0)}**")
        st.write(f"二级 (h2): **{dist.get('h2', 0)}**")
        st.write(f"三级 (h3): **{dist.get('h3', 0)}**")
        st.write(f"四级 (h4): **{dist.get('h4', 0)}**")

    # ── Quality assessment ────────────────────────────────────
    quality = results.get("quality", {})
    if quality:
        st.divider()
        st.subheader("🔍 结构质量评估")
        level = quality.get("level", "medium")
        badge_class = {"high": "quality-high", "medium": "quality-medium", "low": "quality-low"}.get(level, "quality-medium")
        st.markdown(
            f"**综合评级**：<span class='quality-badge {badge_class}'>{level.upper()}</span>",
            unsafe_allow_html=True
        )

        strengths = quality.get("strengths", [])
        weaknesses = quality.get("weaknesses", [])
        suggestions = quality.get("suggestions", [])

        c1, c2 = st.columns(2)
        with c1:
            if strengths:
                st.write("✅ **优点**")
                for s in strengths:
                    st.write(f"• {s}")
        with c2:
            if weaknesses:
                st.write("⚠️ **不足**")
                for w in weaknesses:
                    st.write(f"• {w}")

        if suggestions:
            st.write("💡 **改进建议**")
            for i, s in enumerate(suggestions, 1):
                st.write(f"{i}. {s}")

    # ── Document outline ──────────────────────────────────────
    st.divider()

    if results.get("has_preamble") and results.get("preamble"):
        with st.expander("📝 前言内容"):
            st.write(results["preamble"][:1000])

    if results.get("structure_tree") or results.get("sections"):
        st.subheader("📑 文档大纲")
        tree = results.get("structure_tree", [])
        if tree:
            outline = st.session_state.application_service.structure_analyzer.format_tree_outline(tree)
            render_outline(outline)

    # ── Validation ────────────────────────────────────────────
    validation = results.get("validation", {})
    if validation.get("has_issues"):
        st.divider()
        st.subheader("⚠️ 格式规范检查")
        st.warning(validation.get("summary", ""))
        if validation.get("suggestions"):
            with st.expander("💡 优化建议"):
                for i, s in enumerate(validation["suggestions"], 1):
                    st.write(f"{i}. {s}")
