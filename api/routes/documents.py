"""Document management routes."""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException

from api.schemas import DocumentInfo, DocumentListResponse, DocumentDeleteResponse
from api.dependencies import require_auth, get_history_manager
from src.document_processor import clean_text

router = APIRouter(prefix="/api/documents", tags=["documents"])


document_store = {}


@router.post("/upload", response_model=DocumentInfo, dependencies=[Depends(require_auth)])
async def upload_document(file: UploadFile = File(...)):
    history_mgr = get_history_manager()
    content = await file.read()

    if file.filename and file.filename.endswith(".pdf"):
        import io
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif file.filename and file.filename.endswith(".docx"):
        import io
        from docx import Document as DocxDocument
        doc = DocxDocument(io.BytesIO(content))
        text = "\n".join(p.text for p in doc.paragraphs)
    elif file.filename and file.filename.endswith(".txt") or (file.filename and file.filename.endswith(".md")):
        text = content.decode("utf-8", errors="ignore")
    else:
        try:
            text = content.decode("utf-8", errors="ignore")
        except Exception:
            raise HTTPException(status_code=400, detail="Unsupported file format")

    text = clean_text(text)
    history_mgr.add_entry(
        filename=file.filename or "unknown",
        file_path="",
        file_size=len(content),
        word_count=len(text),
        chunk_count=len(text) // 500 + 1,
    )

    doc_id = history_mgr.history[0].id if history_mgr.history else "unknown"
    document_store[doc_id] = text

    return DocumentInfo(
        id=doc_id,
        filename=file.filename or "unknown",
        file_size=len(content),
        file_size_human=f"{len(content) / 1024:.1f} KB" if len(content) >= 1024 else f"{len(content)} B",
        word_count=len(text),
        processed_at=history_mgr.history[0].processed_at_str if history_mgr.history else "",
    )


@router.get("", response_model=DocumentListResponse, dependencies=[Depends(require_auth)])
def list_documents():
    history_mgr = get_history_manager()
    docs = []
    for entry in history_mgr.get_recent_entries(50):
        docs.append(DocumentInfo(
            id=entry.id,
            filename=entry.filename,
            file_size=entry.file_size,
            file_size_human=entry.file_size_human,
            word_count=entry.word_count,
            processed_at=entry.processed_at_str,
        ))
    return DocumentListResponse(total=len(docs), documents=docs)


@router.get("/{doc_id}", dependencies=[Depends(require_auth)])
def get_document(doc_id: str):
    history_mgr = get_history_manager()
    entry = history_mgr.get_entry_by_id(doc_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "id": entry.id,
        "filename": entry.filename,
        "file_size": entry.file_size,
        "file_size_human": entry.file_size_human,
        "word_count": entry.word_count,
        "processed_at": entry.processed_at_str,
    }


@router.delete("/{doc_id}", response_model=DocumentDeleteResponse, dependencies=[Depends(require_auth)])
def delete_document(doc_id: str):
    history_mgr = get_history_manager()
    if history_mgr.delete_entry(doc_id):
        document_store.pop(doc_id, None)
        return DocumentDeleteResponse(success=True, message="Document deleted")
    raise HTTPException(status_code=404, detail="Document not found")
