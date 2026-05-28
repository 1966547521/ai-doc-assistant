"""Performance benchmarks for key modules.

Run:  pytest tests/benchmark_performance.py -v --durations=0
These use mocks — no sample docs or API keys needed.
"""
import time
import pytest
from unittest.mock import Mock, patch

from src.cache_manager import SemanticCacheManager
from src.document_processor import DocumentProcessor
from src.qa_engine import QAEngine


class TestCachePerformance:
    """Cache hit rate, miss rate, and throughput."""

    @pytest.fixture
    def cache(self):
        c = SemanticCacheManager(cache_dir="cache", max_entries=500)
        c.clear_all()
        return c

    def test_small_string_throughput(self, cache):
        """Measure how many small entries can be set per second."""
        n = 100
        start = time.perf_counter()
        for i in range(n):
            cache.set(f"key{i}", f"value{i}")
        elapsed = time.perf_counter() - start
        ops = n / elapsed
        print(f"\n  S{chr(98)}: {n} 条写入, {elapsed:.3f}s, {ops:.0f} ops/s")

    def test_hit_rate_after_repeated_access(self, cache):
        """Verify 100% hit rate on repeated identical reads."""
        cache.set("same_key", "hello")
        for _ in range(10):
            cache.get("same_key", None)
        stats = cache.get_cache_stats()
        hit_rate = stats["hit_rate"]
        print(f"\n  S{chr(98)}: 10 次重复读取, hit_rate={hit_rate:.1%}")
        assert hit_rate == 1.0, f"Expected 100% hit rate, got {hit_rate}"

    def test_miss_rate_on_unknown_keys(self, cache):
        """Hit rate should be 0 when reading never-written keys."""
        for i in range(10):
            cache.get(f"unknown_{i}", None)
        stats = cache.get_cache_stats()
        hit_rate = stats["hit_rate"]
        print(f"\n  S{chr(98)}: 10 次不存在的读取, hit_rate={hit_rate:.1%}")
        assert hit_rate == 0.0

    def test_lru_eviction_overhead(self, cache):
        """Measure overhead when exceeding max_entries."""
        # Fill to limit
        n = cache.max_entries
        start = time.perf_counter()
        for i in range(n + 50):
            cache.set(f"evict_key_{i}", "x" * 100)
        elapsed = time.perf_counter() - start
        print(f"\n  S{chr(98)}: 超过 max_entries({n}) + 50, {elapsed:.3f}s")
        assert cache.get_cache_stats()["total_entries"] <= cache.max_entries


class TestDocumentProcessorPerformance:
    """Document processing throughput with mocked files."""

    @pytest.fixture
    def processor(self):
        return DocumentProcessor()

    def test_text_split_speed(self, processor):
        """Measure RecursiveCharacterTextSplitter throughput."""
        # ~100 KB of Chinese + English mixed text
        text = ("这是测试文档。" * 500 + "Hello world.\n" * 500) * 10
        assert len(text) > 80000

        start = time.perf_counter()
        chunks = processor.split_text(text)
        elapsed = time.perf_counter() - start

        total_chars = sum(len(c.page_content) for c in chunks)
        speed = total_chars / elapsed if elapsed > 0 else 0
        print(f"\n  S{chr(98)}: {len(text)} chars → {len(chunks)} chunks, "
              f"{elapsed:.4f}s, {speed:.0f} chars/s")

    def test_text_cleanup_speed(self, processor):
        """Measure clean_text throughput."""
        text = ("Normal text.\n" * 1000 +
                "\ufeff" * 100 + "\u200b" * 100 +
                "特殊中文\n" * 1000)

        from src.document_processor import clean_text
        start = time.perf_counter()
        cleaned = clean_text(text)
        elapsed = time.perf_counter() - start
        print(f"\n  S{chr(98)}: {len(text)} chars → {len(cleaned)} chars, "
              f"{elapsed:.4f}s, speed={len(text)/elapsed:.0f} chars/s")


class TestQAPerformance:
    """Mock-based QA pipeline timing (no real API calls)."""

    def test_answer_without_retriever_is_instant(self):
        """Fallback path should return immediately."""
        engine = QAEngine(cache_manager=SemanticCacheManager(cache_dir="cache"))
        start = time.perf_counter()
        for _ in range(10):
            result = engine.answer("test")
        elapsed = time.perf_counter() - start
        avg = elapsed / 10
        print(f"\n  S{chr(98)}: 10 次调用（无 retriever）, avg={avg*1000:.1f}ms")
        assert avg < 0.05, f"Fallback too slow: {avg:.4f}s"

    def test_mock_rag_chain_throughput(self):
        """Measure overhead of QAEngine with mocked retriever."""
        engine = QAEngine(cache_manager=SemanticCacheManager(cache_dir="cache"))
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = [
            Mock(page_content="Relevant content about AI.")
        ]
        engine.set_retriever(mock_retriever)

        class FastRagChain:
            def invoke(self, q): return f"Answer: {q}"
        engine.rag_chain = FastRagChain()

        questions = [f"question_{i}" for i in range(20)]
        start = time.perf_counter()
        results = engine.batch_answer(questions)
        elapsed = time.perf_counter() - start
        avg = elapsed / len(questions)
        print(f"\n  S{chr(98)}: 批量 {len(questions)} 条, "
              f"{elapsed:.3f}s total, {avg*1000:.1f}ms avg")
        assert len(results) == len(questions)

    def test_stream_answer_overhead(self):
        """Measure streaming overhead."""
        engine = QAEngine(cache_manager=SemanticCacheManager(cache_dir="cache"))
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []

        class StreamingChain:
            def stream(self, q):
                for word in ["Part ", "one ", "two ", "three"]:
                    yield word

        engine.set_retriever(mock_retriever)
        engine.rag_chain = StreamingChain()

        start = time.perf_counter()
        chunks = list(engine.stream_answer("test"))
        elapsed = time.perf_counter() - start
        print(f"\n  S{chr(98)}: stream {len(chunks)} chunks, {elapsed*1000:.1f}ms")
        assert len(chunks) == 4


class TestVectorStorePerformance:
    """Mock-based vector store benchmarks."""

    def test_document_id_generation_speed(self):
        """Measure SHA-256 hashing throughput."""
        content = "test document content " * 500  # ~10KB
        from src.vector_store import VectorStoreManager
        with patch.object(VectorStoreManager, '_load_document_ids', return_value=None):
            vs = VectorStoreManager(persist_directory="./chroma_db_test_bench")

        start = time.perf_counter()
        n = 200
        for i in range(n):
            vs._compute_content_hash(f"{content}_{i}")
        elapsed = time.perf_counter() - start
        print(f"\n  S{chr(98)}: {n} 个 hash, {elapsed:.4f}s, "
              f"{n/elapsed:.0f} hash/s")

    def test_dimension_check_speed(self):
        """Measure embedding dimension detection."""
        from src.vector_store import VectorStoreManager
        with patch.object(VectorStoreManager, '_load_document_ids', return_value=None):
            vs = VectorStoreManager(persist_directory="./chroma_db_test_bench")

        mock_emb = Mock()
        mock_emb.embed_query.return_value = [0.1] * 1536
        vs._embeddings = mock_emb

        with patch.object(vs, '_get_collection_dimension', return_value=1536):
            start = time.perf_counter()
            for _ in range(50):
                vs._check_dimension()
            elapsed = time.perf_counter() - start
            per_call = elapsed / 50
            print(f"\n  S{chr(98)}: 50 次 dimension check, avg={per_call*1000:.3f}ms")
