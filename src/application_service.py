"""Application service boundary shared by Streamlit and FastAPI."""

from langchain_core.language_models import BaseChatModel

from src.document_comparer import DocumentComparer
from src.keyword_extractor import KeywordExtractor
from src.report_generator import ReportGenerator
from src.structure_analyzer import StructureAnalyzer
from src.summary_engine import SummaryEngine
from src.translation_engine import TranslationEngine
from src.utils import get_llm


class ApplicationService:
    """Owns document context and analysis engines outside UI session state."""

    def __init__(
        self,
        *,
        summary_engine: SummaryEngine | None = None,
        structure_analyzer: StructureAnalyzer | None = None,
        keyword_extractor: KeywordExtractor | None = None,
        translation_engine: TranslationEngine | None = None,
        report_generator: ReportGenerator | None = None,
        document_comparer: DocumentComparer | None = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        supplied_engines = (
            summary_engine,
            structure_analyzer,
            keyword_extractor,
            translation_engine,
            report_generator,
            document_comparer,
        )
        shared_llm = llm
        if any(engine is None for engine in supplied_engines):
            shared_llm = shared_llm or get_llm()

        self.summary_engine = summary_engine or SummaryEngine(llm=shared_llm)
        self.structure_analyzer = structure_analyzer or StructureAnalyzer(llm=shared_llm)
        self.keyword_extractor = keyword_extractor or KeywordExtractor(llm=shared_llm)
        self.translation_engine = translation_engine or TranslationEngine(llm=shared_llm)
        self.report_generator = report_generator or ReportGenerator(llm=shared_llm)
        self.document_comparer = document_comparer or DocumentComparer(llm=shared_llm)
        self.document_text = ""
        self.document_id: str | None = None

    def set_document(self, text: str, *, document_id: str) -> None:
        if not text.strip():
            raise ValueError("文档内容不能为空")
        self.document_text = text
        self.document_id = document_id

    def clear_document(self) -> None:
        self.document_text = ""
        self.document_id = None
