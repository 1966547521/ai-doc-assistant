"""Document comparison tab for AI Document Assistant."""
import os
import tempfile
import streamlit as st
from src.document_processor import DocumentProcessor
from ui.utils import persist_content


def render_compare_tab():
    """Render the document comparison tab."""
    st.header("🔄 文档对比")

    if not st.session_state.documents_uploaded:
        st.info("👈 请先在左侧侧边栏上传第一个文档，然后在此上传第二个文档进行对比分析。")
        st.markdown("""
        对比功能：
        - 相似度百分比计算
        - 新增/删除/不变行数统计
        - AI 生成的差异摘要
        - 逐行详细差异展示
        """)
        return

    st.write("上传第二个文档进行对比分析")

    compare_file = st.file_uploader(
        "选择要对比的文档",
        type=["pdf", "txt", "md"],
        key="compare_uploader"
    )

    if compare_file:
        processor = DocumentProcessor()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{compare_file.name}", dir=".") as tmp:
            tmp.write(compare_file.getbuffer())
            temp_path = tmp.name

        compare_text = processor.read_document(temp_path)
        os.unlink(temp_path)

        st.session_state.compare_document_text = compare_text
        st.success(f"已加载对比文档: {compare_file.name}")

    if st.session_state.compare_document_text:
        # Show previous comparison if available
        saved = st.session_state.get("_compare_result")
        if saved:
            st.subheader(f"上次对比 · 相似度: {saved.get('similarity', 0)}%")
            st.write(saved.get("summary", ""))
            with st.expander("查看详细差异"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**当前文档**")
                    for line in saved.get("removed", [])[:10]:
                        st.write(f"~~{line}~~")
                with col_b:
                    st.write("**对比文档**")
                    for line in saved.get("added", [])[:10]:
                        st.write(f"+ {line}")
            st.divider()

        if st.button("开始对比"):
            with st.spinner("正在对比文档..."):
                service = st.session_state.application_service
                comparer = service.document_comparer
                text1 = service.document_text
                text2 = st.session_state.compare_document_text

                similarity = comparer.calculate_similarity(text1, text2)

                st.subheader(f"相似度: {similarity}%")

                result = comparer.compare_texts(text1, text2)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("新增行数", result['stats']['added_lines'])
                with col2:
                    st.metric("删除行数", result['stats']['removed_lines'])
                with col3:
                    st.metric("不变行数", result['stats']['unchanged_lines'])

                st.subheader("差异摘要")
                summary = comparer.generate_diff_summary(
                    text1, text2, "当前文档", "对比文档"
                )
                st.write(summary)

                # Save result for persistence
                cmp_data = {"similarity": similarity, "summary": summary,
                            "removed": result['removed'], "added": result['added']}
                persist_content(st.session_state.get("current_session_id"), "_compare_result", cmp_data)

                with st.expander("查看详细差异"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write("**当前文档**")
                        for line in result['removed'][:10]:
                            st.write(f"~~{line}~~")
                    with col_b:
                        st.write("**对比文档**")
                        for line in result['added'][:10]:
                            st.write(f"+ {line}")
