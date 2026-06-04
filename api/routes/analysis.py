"""Analysis endpoints wrapping existing engine layer."""

import hashlib

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
from api.dependencies import require_auth, get_cache_manager
from src.summary_engine import SummaryEngine
from src.qa_engine import QAEngine
from src.structure_analyzer import StructureAnalyzer
from src.keyword_extractor import KeywordExtractor
from src.translation_engine import TranslationEngine
from src.document_comparer import DocumentComparer
from src.report_generator import ReportGenerator

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


def _get_summary_engine():
    return SummaryEngine()


def _get_qa_engine():
    return QAEngine(cache_manager=get_cache_manager())


def _get_structure_analyzer():
    return StructureAnalyzer()


def _get_keyword_extractor():
    return KeywordExtractor()


def _get_translation_engine():
    return TranslationEngine()


def _get_document_comparer():
    return DocumentComparer()


def _get_report_generator():
    return ReportGenerator()


@router.post("/summary", response_model=SummaryResponse, dependencies=[Depends(require_auth)])
def generate_summary(body: SummaryRequest):
    engine = _get_summary_engine()
    result = engine.summarize(
        text=body.text,
        summary_type=body.summary_type,
        language=body.language,
        stream=False,
    )
    summary = result.get("summary", "") if isinstance(result, dict) else str(result)
    return SummaryResponse(summary=summary, summary_type=body.summary_type)


@router.post("/qa", response_model=QAResponse, dependencies=[Depends(require_auth)])
def ask_question(body: QARequest):
    engine = _get_qa_engine()
    context = body.context or ""
    context_hash = hashlib.sha256(context.encode()).hexdigest()[:16] if context else "no_context"

    cache_mgr = get_cache_manager()
    cached = cache_mgr.get_qa_semantic(body.question, context_hash)
    if cached:
        return QAResponse(answer=cached, question=body.question, cached=True)

    if context:
        engine.load_document(context)

    result = engine.ask(
        question=body.question,
        context=context,
        stream=False,
    )
    answer = result.get("answer", "") if isinstance(result, dict) else str(result)
    cache_mgr.cache_qa(body.question, context_hash, answer)
    return QAResponse(answer=answer, question=body.question, cached=False)


@router.post("/structure", response_model=StructureResponse, dependencies=[Depends(require_auth)])
def analyze_structure(body: StructureRequest):
    engine = _get_structure_analyzer()
    result = engine.analyze(body.text)
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
        document_type=result.get("document_type", "unknown"),
        title=result.get("title"),
        total_sections=len(sections),
        sections=sections,
        quality_score=result.get("quality_score"),
    )


@router.post("/keywords", response_model=KeywordsResponse, dependencies=[Depends(require_auth)])
def extract_keywords(body: KeywordsRequest):
    engine = _get_keyword_extractor()
    result = engine.extract(body.text)
    return KeywordsResponse(
        keywords=result.get("keywords", []),
        action_items=result.get("action_items", []),
        topics=result.get("topics", []),
    )


@router.post("/translate", response_model=TranslateResponse, dependencies=[Depends(require_auth)])
def translate_text(body: TranslateRequest):
    engine = _get_translation_engine()
    result = engine.translate(
        text=body.text,
        target_language=body.target_language,
        stream=False,
    )
    translated = result.get("translated_text", "") if isinstance(result, dict) else str(result)
    return TranslateResponse(
        translated_text=translated,
        source_length=len(body.text),
        target_length=len(translated),
        target_language=body.target_language,
    )


@router.post("/compare", response_model=CompareResponse, dependencies=[Depends(require_auth)])
def compare_documents(body: CompareRequest):
    engine = _get_document_comparer()
    result = engine.compare(body.text1, body.text2)
    return CompareResponse(
        similarity_score=result.get("similarity_score", 0.0),
        html_diff=result.get("html_diff", ""),
        common_sections=result.get("common_sections", []),
        stats=result.get("stats", {}),
    )


@router.post("/report", response_model=ReportResponse, dependencies=[Depends(require_auth)])
def generate_report(body: ReportRequest):
    engine = _get_report_generator()
    result = engine.generate(
        text=body.text,
        template=body.template,
    )
    markdown = result.get("markdown", "") if isinstance(result, dict) else str(result)
    return ReportResponse(markdown=markdown, template=body.template)
