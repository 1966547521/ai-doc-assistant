"""Shared UI utility functions."""
from typing import Any
import streamlit as st
from src.session_manager import SessionEntry


SESSION_CONTENT_KEYS = {
    "summary": "_summary_result",
    "translation": "_translation_result",
    "report": "_report_result",
    "compare": "_compare_result",
    "extraction": "extraction_results",
}


def restore_session_content(session: SessionEntry):
    """Load persisted generated_content from session into st.session_state.
    Call this after restoring a session so each tab sees its previous results.
    """
    gc = session.generated_content or {}
    for key, state_key in SESSION_CONTENT_KEYS.items():
        if key in gc and gc[key] is not None:
            st.session_state[state_key] = gc[key]
        elif state_key in st.session_state:
            st.session_state.pop(state_key, None)


def persist_content(session_id: str | None, state_key: str, value: Any):
    """Save generated content to session_state AND persist to session file."""
    st.session_state[state_key] = value
    if session_id:
        # Map state_key back to the storage key
        key_map = {v: k for k, v in SESSION_CONTENT_KEYS.items()}
        content_key = key_map.get(state_key)
        if content_key:
            manager = st.session_state.get("session_manager")
            if manager:
                manager.update_generated_content(session_id, content_key, value)


def render_outline(outline_items: list, indent: int = 0):
    """Render hierarchical outline in Streamlit.

    Args:
        outline_items: List of outline items from format_tree_outline
        indent: Current indentation level
    """
    for item in outline_items:
        level = item.get('level', 1)
        text = item.get('text', '')
        number = item.get('number', '')

        if level == 1:
            st.markdown(f"**{number} {text}**")
        else:
            prefix = "　　" * (level - 2) + "• " if level > 2 else "• "
            st.markdown(f"{prefix}{number} {text}")

        if item.get('children'):
            render_outline(item['children'], indent + 1)

        if item.get('content_preview'):
            preview = item['content_preview'][:150] + "..." if len(item['content_preview']) > 150 else item['content_preview']
            content_indent = "　　" * (level - 1)
            st.markdown(f"{content_indent}<span style='color:#666;font-size:14px'>{preview}</span>", unsafe_allow_html=True)
