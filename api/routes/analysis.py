"""Analysis endpoints wrapping existing engine layer."""

from fastapi import APIRouter, Depends, HTTPException

from api.schemas import (
    SummaryRequest, SummaryResponse,
    QARequest, QAResponse,
    StructureRequest, StructureResponse, SectionInfo,
    KeywordsRequest, KeywordsResponse,
    TranslateRequest, TranslateResponse,
    CompareRequest, CompareResponse,
    ReportRequest, ReportResponse,
)
from api.dependencies import get_application_service, get_document_service, require_auth

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


def _get_summary_engine():
    return get_application_service().summary_engine


def _get_structure_analyzer():
    return get_application_service().structure_analyzer


def _get_keyword_extractor():
    return get_application_service().keyword_extractor


def _get_translation_engine():
    return get_application_service().translation_engine


def _get_document_comparer():
    return get_application_service().document_comparer


def _get_report_generator():
    return get_application_service().report_generator


@router.post("/summary", response_model=SummaryResponse, dependencies=[Depends(require_auth)])
def generate_summary(body: SummaryRequest):
    engine = _get_summary_engine()
    if body.summary_type == "bullet":
        summary = engine.generate_bullet_summary(body.text)
    elif body.summary_type == "executive":
        summary = engine.generate_executive_summary(body.text)
    elif body.summary_type == "qa":
        summary = engine.generate_summary_with_questions(body.text)
    else:
        summary = engine.generate_summary(body.text, length=body.summary_type)
    return SummaryResponse(summary=summary, summary_type=body.summary_type)


@router.post("/qa", response_model=QAResponse, dependencies=[Depends(require_auth)])
def ask_question(body: QARequest):
    try:
        result = get_document_service().ask(body.document_id, body.question)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    return QAResponse(
        answer=str(result.get("answer", "")),
        question=body.question,
        sources=result.get("sources", []),
        citations=result.get("citations", []),
    )


@router.post("/structure", response_model=StructureResponse, dependencies=[Depends(require_auth)])
def analyze_structure(body: StructureRequest):
    engine = _get_structure_analyzer()
    result = engine.analyze_document(body.text)
    sections = [
        SectionInfo(
            title=s.get("title", ""),
            level=s.get("level", 1),
            summary=s.get("summary"),
            start_char=s.get("start_char", 0),
            end_char=s.get("end_char", 0),
        )
        for s in result.get("sections", [])
    ]
    return StructureResponse(
        document_type=result.get("doc_type", "unknown"),
        title=result.get("title"),
        total_sections=len(sections),
        sections=sections,
        quality_score=result.get("quality_score"),
    )


@router.post("/keywords", response_model=KeywordsResponse, dependencies=[Depends(require_auth)])
def extract_keywords(body: KeywordsRequest):
    engine = _get_keyword_extractor()
    return KeywordsResponse(
        keywords=engine.extract_key_terms(body.text),
        action_items=engine.extract_actions(body.text),
        topics=engine.extract_topics(body.text),
    )


@router.post("/translate", response_model=TranslateResponse, dependencies=[Depends(require_auth)])
def translate_text(body: TranslateRequest):
    engine = _get_translation_engine()
    result = engine.translate(text=body.text, target_lang=body.target_language)
    translated = result.get("translation", "") if isinstance(result, dict) else str(result)
    return TranslateResponse(
        translated_text=translated,
        source_length=len(body.text),
        target_length=len(translated),
        target_language=body.target_language,
    )


@router.post("/compare", response_model=CompareResponse, dependencies=[Depends(require_auth)])
def compare_documents(body: CompareRequest):
    engine = _get_document_comparer()
    result = engine.compare_texts(body.text1, body.text2)
    return CompareResponse(
        similarity_score=float(result.get("similarity", 0.0)),
        html_diff="",
        common_sections=[],
        stats=result.get("stats", {}),
    )


@router.post("/report", response_model=ReportResponse, dependencies=[Depends(require_auth)])
def generate_report(body: ReportRequest):
    engine = _get_report_generator()
    markdown = engine.generate_markdown_report(body.text, template=body.template)
    return ReportResponse(markdown=markdown, template=body.template)
