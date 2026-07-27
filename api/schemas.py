"""Pydantic schemas for FastAPI request/response models."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Auth ───────────────────────────────────────────

class LoginRequest(BaseModel):
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── Document ───────────────────────────────────────

class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_size: int
    file_size_human: str
    word_count: int
    processed_at: str


class DocumentListResponse(BaseModel):
    total: int
    documents: List[DocumentInfo]


class DocumentDeleteResponse(BaseModel):
    success: bool
    message: str


# ── Analysis ───────────────────────────────────────

class SummaryRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Document text to summarize")
    summary_type: str = Field("detailed", description="Summary type: short, detailed, bullet, executive, qa")
    language: str = Field("zh", description="Output language")


class SummaryResponse(BaseModel):
    summary: str
    summary_type: str


class QARequest(BaseModel):
    question: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class QAResponse(BaseModel):
    answer: str
    question: str
    cached: bool = False
    sources: List[str] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)


class StructureRequest(BaseModel):
    text: str = Field(..., min_length=1)


class SectionInfo(BaseModel):
    title: str
    level: int
    summary: Optional[str] = None
    start_char: int
    end_char: int


class StructureResponse(BaseModel):
    document_type: str
    title: Optional[str] = None
    total_sections: int
    sections: List[SectionInfo]
    quality_score: Optional[float] = None


class KeywordsRequest(BaseModel):
    text: str = Field(..., min_length=1)


class KeywordsResponse(BaseModel):
    keywords: List[str]
    action_items: List[str]
    topics: List[str]


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    target_language: str = Field("en", description="Target language code")


class TranslateResponse(BaseModel):
    translated_text: str
    source_length: int
    target_length: int
    target_language: str


class CompareRequest(BaseModel):
    text1: str = Field(..., min_length=1, description="First document text")
    text2: str = Field(..., min_length=1, description="Second document text")


class CompareResponse(BaseModel):
    similarity_score: float
    html_diff: str
    common_sections: List[str]
    stats: Dict[str, Any]


class ReportRequest(BaseModel):
    text: str = Field(..., min_length=1)
    template: str = Field("comprehensive", description="Report template: comprehensive, executive, technical")


class ReportResponse(BaseModel):
    markdown: str
    template: str


# ── Admin ──────────────────────────────────────────

class SystemStats(BaseModel):
    redis: Dict[str, Any]
    cache: Dict[str, Any]
    sessions_count: int
    history_count: int


class SessionInfo(BaseModel):
    id: str
    name: str
    created_at: str
    updated_at: str
    document_count: int
    word_count: str


class SessionListResponse(BaseModel):
    total: int
    sessions: List[SessionInfo]


class MessageResponse(BaseModel):
    success: bool
    message: str
