"""Document management routes backed by the shared document service."""

import os
import tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException

from api.schemas import DocumentInfo, DocumentListResponse, DocumentDeleteResponse
from api.dependencies import require_auth, get_document_service, get_history_manager

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentInfo, dependencies=[Depends(require_auth)])
async def upload_document(file: UploadFile = File(...)):
    history_mgr = get_history_manager()
    content = await file.read()
    filename = file.filename or "unknown"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(filename).suffix
        ) as temporary:
            temporary.write(content)
            temp_path = Path(temporary.name)
        record = get_document_service().ingest_path(temp_path, filename=filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path and temp_path.exists():
            os.unlink(temp_path)

    history_mgr.add_entry(
        filename=filename,
        file_path="",
        file_size=len(content),
        word_count=len(record.text),
        chunk_count=len(record.chunks),
        entry_id=record.id,
    )

    return DocumentInfo(
        id=record.id,
        filename=filename,
        file_size=len(content),
        file_size_human=f"{len(content) / 1024:.1f} KB" if len(content) >= 1024 else f"{len(content)} B",
        word_count=len(record.text),
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
    record = get_document_service().get(doc_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "id": record.id,
        "filename": record.filename,
        "word_count": len(record.text),
    }


@router.delete("/{doc_id}", response_model=DocumentDeleteResponse, dependencies=[Depends(require_auth)])
def delete_document(doc_id: str):
    history_mgr = get_history_manager()
    if get_document_service().delete(doc_id):
        history_mgr.delete_entry(doc_id)
        return DocumentDeleteResponse(success=True, message="Document deleted")
    raise HTTPException(status_code=404, detail="Document not found")
