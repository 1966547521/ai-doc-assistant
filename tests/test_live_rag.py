"""Opt-in acceptance test using real configured APIs and a real Word document."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.document_processor import DocumentProcessor
from src.qa_engine import QAEngine
from src.vector_store import VectorStoreManager


pytestmark = pytest.mark.live_api


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_API") != "1",
    reason="set RUN_LIVE_API=1 to allow real API cost",
)
def test_real_docx_rag_returns_traceable_citation(tmp_path):
    load_dotenv()
    source = Path(__file__).parent / "sample_docs" / "sample_zh.docx"
    processed = DocumentProcessor().process_document(str(source))
    store = VectorStoreManager(
        persist_directory=str(tmp_path / "chroma"),
        index_id=processed["chunks"][0].metadata["document_id"],
    )
    store.add_documents(processed["chunks"], incremental=False)
    engine = QAEngine()
    engine.set_retriever(store.get_store().as_retriever(search_kwargs={"k": 3}))

    result = engine.answer("这份文档介绍的核心项目是什么？")

    assert result["answer"].strip()
    assert result["citations"]
    assert all(item["source_file"] == "sample_zh.docx" for item in result["citations"])
