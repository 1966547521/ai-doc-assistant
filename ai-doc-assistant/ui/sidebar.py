"""Sidebar components for AI Document Assistant."""
import os
import tempfile
import streamlit as st
from src.document_processor import DocumentProcessor
from src.prompt_manager import prompt_manager
from src.logger import get_logger
from ui.utils import restore_session_content

logger = get_logger("ui.sidebar")


def render_sidebar():
    """Render the complete sidebar with upload, nav, sessions, cache, and admin."""
    # ── Brand logo area ──
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:8px 0 16px 0;border-bottom:1px solid #e8edf2;margin-bottom:8px;">
        <div style="
            width:36px;height:36px;border-radius:10px;
            background:linear-gradient(135deg, #4a6fa5, #7ba5d1);
            display:flex;align-items:center;justify-content:center;
            font-size:18px;
        ">📚</div>
        <div>
            <div style="font-weight:700;font-size:1rem;color:#2d3748;">AI 文档助手</div>
            <div style="font-size:0.7rem;color:#8899aa;">智能分析 · 高效办公</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.header("📁 文档上传")

    uploaded_files = st.file_uploader(
        "拖拽文件到此处或点击选择",
        type=["pdf", "docx", "xlsx", "pptx", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.subheader(f"已选择 {len(uploaded_files)} 个文件")
        for _i, file in enumerate(uploaded_files):
            file_size = f"{file.size / 1024:.1f} KB" if file.size < 1024 * 1024 else f"{file.size / (1024 * 1024):.1f} MB"
            st.write(f"• {file.name} ({file_size})")

    use_incremental = st.checkbox(
        "启用增量更新",
        value=True,
        help="跳过已上传的重复文档片段"
    )

    if st.button("开始处理"):
        _handle_processing(uploaded_files, use_incremental)

    if st.button("清除索引"):
        _handle_clear_index()

    st.divider()

    # Feature navigation — always accessible in sidebar, no scrolling needed
    st.header("🧭 功能导航")
    feature_names = [
        "🧠 AI 助手", "📝 文档摘要", "🏗️ 结构分析",
        "🔍 实体提取", "🌍 文档翻译", "📊 报告生成", "🔄 文档对比"
    ]
    feature_map = {name: i for i, name in enumerate(feature_names)}
    active_label = st.radio(
        "选择功能",
        options=feature_names,
        index=feature_map.get(st.session_state.get("active_tab"), 0),
        key="feature_nav",
        label_visibility="collapsed",
        on_change=lambda: setattr(st.session_state, "active_tab", st.session_state.feature_nav)
    )
    if active_label != st.session_state.get("active_tab"):
        st.session_state.active_tab = active_label

    st.divider()
    _render_session_history()
    st.divider()
    _render_cache_status()
    st.divider()
    _render_admin_panel()


def _handle_processing(uploaded_files, use_incremental):
    """Handle document processing workflow."""
    if not uploaded_files:
        logger.warning("No files uploaded for processing")
        st.warning("请先上传文档")
        return

    logger.info(f"Starting document processing for {len(uploaded_files)} files")

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_steps = len(uploaded_files) + 3
    current_step = 0

    processor = DocumentProcessor()
    all_documents = []
    full_text = ""

    for _idx, file in enumerate(uploaded_files):
        current_step += 1
        status_text.text(f"📄 正在读取文件: {file.name}")
        progress_bar.progress(current_step / total_steps)

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}", dir=".") as tmp:
                tmp.write(file.getbuffer())
                temp_path = tmp.name

            text = processor.read_document(temp_path)
            full_text += text + "\n\n"
            documents = processor.split_text(text)
            all_documents.extend(documents)

            os.unlink(temp_path)
            logger.info(f"Processed file: {file.name}, {len(text)} characters")

        except Exception as e:
            logger.error(f"Error processing file {file.name}: {str(e)}", exc_info=True)
            st.error(f"处理文件 {file.name} 时出错: {str(e)}")
            continue

    current_step += 1
    status_text.text("🔄 正在向量化文档...")
    progress_bar.progress(current_step / total_steps)

    try:
        st.session_state.current_document_text = full_text

        result = st.session_state.vector_store.add_documents(
            all_documents,
            incremental=use_incremental
        )

        retriever = st.session_state.vector_store.vector_store.as_retriever()
        st.session_state.qa_engine.set_retriever(retriever)
        st.session_state.qa_engine.set_context_snapshot(full_text[:5000])
        st.session_state.documents_uploaded = True
        logger.info(f"Vectorization completed: {result['added']} added, {result['skipped']} skipped")

    except Exception as e:
        logger.error(f"Error during vectorization: {str(e)}", exc_info=True)
        st.error(f"向量化过程中出错: {str(e)}")

    current_step += 1
    status_text.text("🧠 正在分析文档结构...")
    progress_bar.progress(current_step / total_steps)

    try:
        analyzer = st.session_state.structure_analyzer
        st.session_state.analysis_results = analyzer.analyze_document(full_text)
        logger.info("Document structure analysis completed")

    except Exception as e:
        logger.error(f"Error during structure analysis: {str(e)}", exc_info=True)
        st.error(f"结构分析过程中出错: {str(e)}")

    current_step += 1
    progress_bar.progress(1.0)
    status_text.text("✅ 处理完成!")

    session_name = ", ".join([file.name for file in uploaded_files])
    if len(session_name) > 30:
        session_name = session_name[:30] + "..."

    documents_info = [
        {
            "filename": file.name,
            "file_size": file.size,
            "file_path": f"./uploads/{file.name}"
        }
        for file in uploaded_files
    ]

    session = st.session_state.session_manager.create_session(
        name=session_name,
        documents=documents_info,
        document_text=full_text,
        analysis_results=st.session_state.analysis_results
    )

    st.session_state.current_session_id = session.id
    st.session_state.memory_manager.clear_history()

    doc_count = st.session_state.vector_store.get_document_count()
    message = f"成功处理 {len(uploaded_files)} 个文档"
    message += f"，添加 {result['added']} 个片段"
    if use_incremental and result['skipped'] > 0:
        message += f" (跳过 {result['skipped']} 个重复片段)"
    message += f" (总计: {doc_count} 个片段)"
    st.success(message)
    logger.info(f"Document processing completed successfully: {message}")


def _handle_clear_index():
    """Clear all indexes and caches."""
    logger.info("Clearing all indexes and caches")

    st.session_state.vector_store.clear_store()
    st.session_state.cache_manager.clear_all()
    st.session_state.memory_manager.clear_history()
    st.session_state.documents_uploaded = False
    st.session_state.current_document_text = ""
    st.session_state.analysis_results = {}
    st.session_state.current_session_id = None
    # Clear generated results
    for key in ("_summary_result", "_translation_result", "_report_result", "_compare_result", "extraction_results"):
        st.session_state.pop(key, None)

    st.info("索引已清除")
    logger.info("Indexes and caches cleared successfully")


def _render_session_history():
    """Render session history section."""
    st.header("📜 会话历史")

    search_query = st.text_input("搜索会话", placeholder="输入会话名称...", key="session_search")

    # Show any pending delete/recover feedback from previous action
    if st.session_state.get("_pending_feedback"):
        fb = st.session_state.pop("_pending_feedback")
        if fb["type"] == "deleted":
            st.success(fb["msg"])
        elif fb["type"] == "restored":
            st.success(fb["msg"])

    all_sessions = st.session_state.session_manager.get_all_sessions()

    if search_query:
        filtered_sessions = st.session_state.session_manager.search_by_name(search_query)
    else:
        filtered_sessions = all_sessions[:10]

    if filtered_sessions:
        for session in filtered_sessions:
            # Card-like row with session info + popover menu
            with st.container():
                st.markdown(
                    """
                    <style>
                    .session-card {{
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        padding: 8px 12px;
                        border: 1px solid rgba(255,255,255,0.08);
                        border-radius: 8px;
                        margin-bottom: 8px;
                        background: rgba(255,255,255,0.03);
                        transition: background 0.15s ease;
                    }}
                    .session-card:hover {{
                        background: rgba(255,255,255,0.06);
                    }}
                    .session-name {{
                        color: #E2E8F0;
                        font-weight: 500;
                        font-size: 0.9rem;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        white-space: nowrap;
                        flex: 1;
                    }}
                    .session-meta {{
                        color: #8899AA;
                        font-size: 0.75rem;
                        margin-top: 2px;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                col_info, col_menu = st.columns([5, 1])

                with col_info:
                    with st.expander(f"📄 {session.name}"):
                        c1, c2 = st.columns(2)
                        c1.write(f"📅 创建: {session.created_at_str}")
                        c2.write(f"📊 更新: {session.updated_at_str}")
                        st.write(f"📁 文档数: {session.document_count}")
                        st.write(f"📝 字数: {session.word_count_human}")
                        st.write(f"💬 对话数: {len(session.chat_history)}")

                with col_menu:
                    with st.popover("⋯", use_container_width=False):
                        if st.button("🔄 恢复会话", key=f"restore_{session.id}", use_container_width=True):
                            _on_restore(session)
                        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                        if st.button("🗑️ 删除", key=f"delete_{session.id}", use_container_width=True):
                            _on_delete(session.id, session.name)
    else:
        st.info("暂无会话记录")

    if all_sessions and st.button("🗑️ 清空全部会话", key="clear_sessions"):
        st.session_state.session_manager.clear_sessions()
        # Clean up all agent session caches
        for key in list(st.session_state.keys()):
            if key.startswith("_agent_session_") or key.startswith("_agent_messages_"):
                del st.session_state[key]
        st.session_state.current_session_id = None
        st.session_state.documents_uploaded = False
        st.session_state.current_document_text = ""
        st.session_state.analysis_results = {}
        st.session_state.memory_manager.clear_history()
        st.session_state.vector_store.clear_store()
        st.session_state.pop("_summary_result", None)
        st.session_state.pop("_translation_result", None)
        st.session_state.pop("extraction_results", None)
        st.session_state._pending_feedback = {"type": "deleted", "msg": "会话记录与页面数据已清空"}
        st.rerun()


def _on_delete(session_id: str, session_name: str):
    """Callback for deleting a session."""
    st.session_state.session_manager.delete_session(session_id)
    if st.session_state.current_session_id == session_id:
        st.session_state.current_session_id = None
        st.session_state.documents_uploaded = False
        st.session_state.current_document_text = ""
        st.session_state.analysis_results = {}
        st.session_state.memory_manager.clear_history()
        st.session_state.vector_store.clear_store()
        # Clean up agent session
        st.session_state.pop(f"_agent_session_{session_id}", None)
        st.session_state.pop(f"_agent_messages_{session_id}", None)
    else:
        # Clean up stale agent cache even if not current session
        st.session_state.pop(f"_agent_session_{session_id}", None)
        st.session_state.pop(f"_agent_messages_{session_id}", None)
    st.session_state._pending_feedback = {"type": "deleted", "msg": f"已删除: {session_name}"}
    st.rerun()


def _on_restore(session):
    """Callback for restoring a session."""
    _restore_session(session)
    # Clear stale agent cache so new session gets fresh AgentSession
    st.session_state.pop(f"_agent_session_{session.id}", None)
    st.rerun()


def _restore_session(session):
    """Restore a previous session — re-indexes docs and loads all generated content."""
    st.session_state.current_session_id = session.id
    st.session_state.current_document_text = session.document_text
    st.session_state.analysis_results = session.analysis_results

    processor = DocumentProcessor()
    documents = processor.split_text(session.document_text)
    st.session_state.vector_store.clear_store()
    st.session_state.vector_store.add_documents(documents, incremental=False)
    retriever = st.session_state.vector_store.vector_store.as_retriever()
    st.session_state.qa_engine.set_retriever(retriever)
    st.session_state.qa_engine.set_context_snapshot(session.document_text[:5000])
    st.session_state.documents_uploaded = True

    st.session_state.memory_manager.clear_history()
    for msg in session.chat_history:
        st.session_state.memory_manager.add_message(msg["role"], msg["content"])

    # Restore all previously generated content (summary, translation, report, etc.)
    restore_session_content(session)

    st.session_state._pending_feedback = {"type": "restored", "msg": f"已恢复: {session.name}"}


def _render_cache_status():
    """Render cache status section."""
    st.header("⚙️ 缓存状态")
    cache_stats = st.session_state.cache_manager.get_cache_stats()
    col1, col2 = st.columns(2)
    col1.metric("条目", cache_stats['total_entries'])
    col2.metric("命中率", f"{cache_stats['hit_rate'] * 100:.1f}%")
    st.caption(f"缓存大小: {cache_stats['total_size_human']}")

    if st.button("清除缓存"):
        st.session_state.cache_manager.clear_all()
        st.info("缓存已清除")


def _render_admin_panel():
    """Render admin panel for prompt management."""
    if "admin_expanded" not in st.session_state:
        st.session_state.admin_expanded = False

    admin_icon = "🔧" if st.session_state.get("admin_authenticated", False) else "⚙️"
    if st.button(admin_icon, key="admin_btn", help="管理员入口"):
        st.session_state.admin_expanded = not st.session_state.admin_expanded

    if not st.session_state.admin_expanded:
        return

    if not st.session_state.get("admin_authenticated", False):
        admin_password = st.text_input(
            "",
            type="password",
            placeholder="请输入密码",
            key="admin_password",
            label_visibility="collapsed"
        )

        expected_password = os.getenv("ADMIN_PASSWORD", "")
        if expected_password and admin_password == expected_password:
            st.session_state.admin_authenticated = True
            st.success("✓ 已认证")
        elif admin_password:
            st.error("密码错误")
    else:
        st.success("✓ 管理员已认证")
        st.header("📝 提示词管理")
        st.write("提示词文件位于 `prompts/` 目录")
        if st.button("重新加载提示词", key="reload_prompts"):
            prompt_manager.reload()
            st.info("提示词已重新加载")

        available_prompts = prompt_manager.list_prompts()
        st.write(f"可用提示词: {len(available_prompts)} 个")
        for p in available_prompts[:5]:
            st.write(f"- {p}")
        if len(available_prompts) > 5:
            st.write(f"... 还有 {len(available_prompts) - 5} 个")

        if st.button("退出管理员模式", key="admin_logout"):
            st.session_state.admin_authenticated = False
            st.session_state.admin_expanded = False
            st.info("已退出管理员模式")
