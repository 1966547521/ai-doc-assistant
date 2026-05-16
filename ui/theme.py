"""Arctic Frost theme — clean, crisp, professional.

Steel Blue   #4a6fa5 — primary accent (buttons, links)
Ice Blue     #d4e4f7 — subtle highlights, card tint
Silver       #c0c0c0 — borders, dividers
Crisp White  #fafafa — page background
White        #ffffff — card / sidebar surface
Dark Slate   #2d3748 — text
"""

import streamlit as st

CSS = r"""
/* ===== BRAND LOGO ===== */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0 16px 0;
    border-bottom: 1px solid #e8edf2;
    margin-bottom: 8px;
}
.sidebar-brand-icon {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    background: linear-gradient(135deg, #4a6fa5, #7ba5d1);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}
.sidebar-brand-name {
    font-weight: 700;
    font-size: 1rem;
    color: #2d3748;
}
.sidebar-brand-tagline {
    font-size: 0.7rem;
    color: #8899aa;
}
.title-accent-bar {
    width: 48px;
    height: 4px;
    background: linear-gradient(90deg, #4a6fa5, #7ba5d1);
    border-radius: 2px;
    margin-bottom: 8px;
}

/* ===== SIDEBAR — airy light ===== */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #4a6fa5 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #e8edf2 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #4a5568 !important;
}

/* Sidebar feature nav — selected item */
[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"] {
    background-color: #4a6fa5 !important;
    color: #fff !important;
    border-radius: 6px !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] button[kind="primary"] {
    background-color: #4a6fa5 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    color: #fff !important;
}

/* Sidebar inputs */
[data-testid="stSidebar"] input {
    border: 1px solid #d4e4f7 !important;
    border-radius: 8px !important;
    background: #fafafa !important;
}

/* Sidebar file uploader */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    border: 2px dashed #d4e4f7 !important;
    border-radius: 12px !important;
    background: #fafbfd !important;
}

/* Sidebar expander cards */
[data-testid="stSidebar"] details {
    border: 1px solid #e8edf2 !important;
    border-radius: 8px !important;
    margin-bottom: 4px !important;
}

/* ---- MAIN CONTENT ---- */
div[data-testid="stExpander"] {
    border: 1px solid #e8edf2 !important;
    border-radius: 10px !important;
}

button[kind="primary"] {
    background-color: #4a6fa5 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
button[kind="primary"]:hover {
    background-color: #3b5c8a !important;
    box-shadow: 0 2px 10px rgba(74,111,165,0.25);
}

/* ---- CHAT ---- */
[data-testid="stChatMessage"][aria-label*="user"] {
    border-left: 3px solid #4a6fa5 !important;
    padding-left: 12px !important;
    background: rgba(74,111,165,0.04) !important;
}
[data-testid="stChatMessage"][aria-label*="assistant"] {
    border-left: 3px solid #c0c0c0 !important;
    padding-left: 12px !important;
}

.stChatInput textarea {
    border: 2px solid #d4e4f7 !important;
    border-radius: 10px !important;
}
.stChatInput textarea:focus {
    border-color: #4a6fa5 !important;
    box-shadow: 0 0 0 3px rgba(74,111,165,0.12) !important;
}

/* ---- METRICS ---- */
[data-testid="stMetricValue"] {
    color: #4a6fa5 !important;
}

/* ---- CUSTOM ---- */
.insight-card {
    background: linear-gradient(135deg, rgba(74,111,165,0.04), rgba(212,228,247,0.15));
    padding: 14px 18px;
    border-radius: 10px;
    border-left: 4px solid #4a6fa5;
    margin-bottom: 1rem;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #2d3748;
}

.quality-high   { display:inline-block;padding:4px 14px;border-radius:20px;font-weight:700;font-size:0.8rem;text-transform:uppercase;background:rgba(72,187,120,0.12);color:#48bb78; }
.quality-medium  { display:inline-block;padding:4px 14px;border-radius:20px;font-weight:700;font-size:0.8rem;text-transform:uppercase;background:rgba(237,137,54,0.12);color:#ed8936; }
.quality-low    { display:inline-block;padding:4px 14px;border-radius:20px;font-weight:700;font-size:0.8rem;text-transform:uppercase;background:rgba(245,101,101,0.12);color:#f56565; }

.qa-messages-wrapper {
    max-height: 55vh;
    overflow-y: auto;
    padding-right: 8px;
    margin-bottom: 12px;
}

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #d4e4f7; border-radius: 3px; }
"""


def inject_theme():
    """Inject custom CSS for Arctic Frost."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
