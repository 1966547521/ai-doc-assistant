"""Streamlit application for AI Document Assistant.
Main entry point with session initialization and feature routing.
"""
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from src.vector_store import VectorStoreManager
from src.qa_engine import QAEngine
from src.memory_manager import MemoryManager
from src.cache_manager import SemanticCacheManager
from src.session_manager import SessionManager
from src.application_service import ApplicationService
from src.logger import get_logger
from ui.theme import inject_theme
from ui.sidebar import render_sidebar
from ui.agent_chat import render_agent_chat
from ui.summary_tab import render_summary_tab
from ui.structure_tab import render_structure_tab
from ui.entity_tab import render_entity_tab
from ui.translation_tab import render_translation_tab
from ui.report_tab import render_report_tab
from ui.compare_tab import render_compare_tab

logger = get_logger("app")
load_dotenv()

st.set_page_config(page_title="AI 智能文档助手", page_icon="🧠", layout="wide")
inject_theme()
logger.info("AI Document Assistant application starting")


def init_session_state():
    """Initialize session state variables."""
    if st.session_state.get("_initialized"):
        return

    with st.spinner("正在初始化引擎，请稍候..."):
        if "application_service" not in st.session_state:
            try:
                st.session_state.application_service = ApplicationService()
            except RuntimeError as exc:
                st.error(f"API 配置不可用：{exc}")
                st.info("请按照 .env.example 分别配置 Chat 与 Embedding API 后重新启动。")
                st.stop()
        service = st.session_state.application_service
        defaults = {
            "vector_store": VectorStoreManager,
            "cache_manager": SemanticCacheManager,
            "qa_engine": lambda: QAEngine(cache_manager=st.session_state.cache_manager),
            "memory_manager": MemoryManager,
            "summary_engine": lambda: service.summary_engine,
            "structure_analyzer": lambda: service.structure_analyzer,
            "keyword_extractor": lambda: service.keyword_extractor,
            "translation_engine": lambda: service.translation_engine,
            "report_generator": lambda: service.report_generator,
            "document_comparer": lambda: service.document_comparer,
            "session_manager": SessionManager,
        }

        for key, factory in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = factory() if callable(factory) else factory

        scalar_defaults = {
            "current_session_id": None,
            "compare_document_text": "",
            "documents_uploaded": False,
            "current_document_text": "",
            "analysis_results": {},
            "active_tab": "🧠 AI 助手",
        }
        for key, value in scalar_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    st.session_state["_initialized"] = True


# Route to features
FEATURE_ROUTES = {
    "🧠 AI 助手": render_agent_chat,
    "📝 文档摘要": render_summary_tab,
    "🏗️ 结构分析": render_structure_tab,
    "🔍 实体提取": render_entity_tab,
    "🌍 文档翻译": render_translation_tab,
    "📊 报告生成": render_report_tab,
    "🔄 文档对比": render_compare_tab,
}


def main():
    init_session_state()

    # ── Title with decorative accent bar ──
    st.markdown("""
    <div style="margin-bottom:8px;">
        <div style="width:48px;height:4px;background:linear-gradient(90deg,#4a6fa5,#7ba5d1);border-radius:2px;margin-bottom:8px;"></div>
    </div>
    """, unsafe_allow_html=True)
    st.title("🧠 AI 智能文档助手")
    st.caption("上传文档，智能分析与问答 · 支持 PDF / DOCX / PPTX / XLSX / TXT / MD")

    with st.sidebar:
        render_sidebar()

    # Scroll to top when switching features
    prev_tab = st.session_state.get("_prev_tab", "")
    current_tab = st.session_state.get("active_tab", "🧠 AI 助手")
    if prev_tab != current_tab:
        st.session_state._prev_tab = current_tab
        components.html("<script>window.scrollTo(0, 0);</script>", height=0)

    # Render only the active feature
    render_fn = FEATURE_ROUTES.get(current_tab, render_agent_chat)
    render_fn()


if __name__ == "__main__":
    main()
