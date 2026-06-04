"""Admin management routes."""

from fastapi import APIRouter, Depends, HTTPException

from api.schemas import (
    SystemStats, SessionInfo, SessionListResponse, MessageResponse,
)
from api.dependencies import (
    require_auth, get_cache_manager,
    get_session_manager, get_history_manager,
)

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/stats", response_model=SystemStats, dependencies=[Depends(require_auth)])
def get_stats():
    cache_mgr = get_cache_manager()
    session_mgr = get_session_manager()
    history_mgr = get_history_manager()

    return SystemStats(
        redis={"available": False},
        cache=cache_mgr.get_cache_stats(),
        sessions_count=len(session_mgr.get_all_sessions()),
        history_count=len(history_mgr.get_all_entries()),
    )


@router.post("/cache/clear", response_model=MessageResponse, dependencies=[Depends(require_auth)])
def clear_cache():
    cache_mgr = get_cache_manager()
    cache_mgr.clear_all()
    return MessageResponse(success=True, message="Cache cleared")


@router.get("/sessions", response_model=SessionListResponse, dependencies=[Depends(require_auth)])
def list_sessions():
    session_mgr = get_session_manager()
    sessions = [
        SessionInfo(
            id=s.id,
            name=s.name,
            created_at=s.created_at_str,
            updated_at=s.updated_at_str,
            document_count=s.document_count,
            word_count=s.word_count_human,
        )
        for s in session_mgr.get_all_sessions()
    ]
    return SessionListResponse(total=len(sessions), sessions=sessions)


@router.delete("/sessions/{session_id}", response_model=MessageResponse, dependencies=[Depends(require_auth)])
def delete_session(session_id: str):
    session_mgr = get_session_manager()
    if session_mgr.delete_session(session_id):
        return MessageResponse(success=True, message="Session deleted")
    raise HTTPException(status_code=404, detail="Session not found")


@router.post("/sessions/clear", response_model=MessageResponse, dependencies=[Depends(require_auth)])
def clear_sessions():
    session_mgr = get_session_manager()
    session_mgr.clear_sessions()
    return MessageResponse(success=True, message="All sessions cleared")


@router.get("/health", dependencies=[Depends(require_auth)])
def health_check():
    return {"status": "ok"}
