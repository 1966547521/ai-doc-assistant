"""Document ingestion, isolated indexing and traceable RAG orchestration."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from src.document_processor import DocumentProcessor
from src.qa_engine import QAEngine
from src.vector_store import VectorStoreManager
from src.runtime_paths import CHROMA_DIR, ensure_runtime_directories


@dataclass
class DocumentRecord:
    id: str
    filename: str
    text: str
    chunks: list
    qa_engine: QAEngine


class DocumentService:
    def __init__(self, persist_directory: str = str(CHROMA_DIR)) -> None:
        ensure_runtime_directories()
        self.persist_directory = persist_directory
        self._records: dict[str, DocumentRecord] = {}

    @staticmethod
    def document_id_for(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def ingest_path(self, path: Path, *, filename: str) -> DocumentRecord:
        extension = Path(filename).suffix.lower()
        if extension not in {".pdf", ".docx"}:
            raise ValueError("仅支持 PDF 和 DOCX 文档")
        content = path.read_bytes()
        if len(content) > 20 * 1024 * 1024:
            raise ValueError("文档大小不能超过 20 MB")
        document_id = self.document_id_for(content)
        processed = DocumentProcessor(enable_semantic_chunking=True).process_document(
            str(path)
        )
        if not processed["text"].strip():
            raise ValueError("未提取到可索引文本；扫描版 PDF 需要先执行 OCR")
        for chunk in processed["chunks"]:
            chunk.metadata["document_id"] = document_id
            chunk.metadata["source_file"] = filename
            suffix = chunk.metadata.get("chunk_id", "").split(":")[-1]
            chunk.metadata["chunk_id"] = f"{document_id}:{suffix}"
        store = VectorStoreManager(
            persist_directory=self.persist_directory, index_id=document_id
        )
        store.add_documents(processed["chunks"], incremental=True)
        qa_engine = QAEngine()
        qa_engine.set_retriever(store.get_store().as_retriever(search_kwargs={"k": 4}))
        qa_engine.set_context_snapshot(document_id)
        record = DocumentRecord(
            id=document_id,
            filename=filename,
            text=processed["text"],
            chunks=processed["chunks"],
            qa_engine=qa_engine,
        )
        self._records[document_id] = record
        return record

    def get(self, document_id: str) -> DocumentRecord | None:
        return self._records.get(document_id)

    def ask(self, document_id: str, question: str) -> dict:
        record = self.get(document_id)
        if record is None:
            raise KeyError(document_id)
        return record.qa_engine.answer(question)

    def delete(self, document_id: str) -> bool:
        return self._records.pop(document_id, None) is not None
