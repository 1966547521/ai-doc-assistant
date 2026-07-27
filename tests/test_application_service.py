from unittest.mock import Mock, patch

from src.application_service import ApplicationService


def test_document_context_lives_in_application_service():
    service = ApplicationService(
        summary_engine=Mock(),
        structure_analyzer=Mock(),
        keyword_extractor=Mock(),
        translation_engine=Mock(),
        report_generator=Mock(),
        document_comparer=Mock(),
    )

    service.set_document("正文", document_id="doc-1")

    assert service.document_text == "正文"
    assert service.document_id == "doc-1"
    service.clear_document()
    assert service.document_text == ""


def test_default_engines_share_one_llm_client():
    llm = Mock()

    with (
        patch("src.application_service.get_llm", return_value=llm) as get_llm,
        patch("src.application_service.SummaryEngine") as summary,
        patch("src.application_service.StructureAnalyzer") as structure,
        patch("src.application_service.KeywordExtractor") as keywords,
        patch("src.application_service.TranslationEngine") as translation,
        patch("src.application_service.ReportGenerator") as report,
        patch("src.application_service.DocumentComparer") as comparer,
    ):
        ApplicationService()

    get_llm.assert_called_once_with()
    summary.assert_called_once_with(llm=llm)
    structure.assert_called_once_with(llm=llm)
    keywords.assert_called_once_with(llm=llm)
    translation.assert_called_once_with(llm=llm)
    report.assert_called_once_with(llm=llm)
    comparer.assert_called_once_with(llm=llm)
