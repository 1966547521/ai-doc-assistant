"""
Manual end-to-end performance benchmark for ai-doc-assistant.

This script measures real pipeline performance — you need:
  1. API keys in .env (DashScope / OpenAI / etc.)
  2. Sample documents in tests/sample_docs/ (or specify your own paths)

Usage:
    python benchmark.py                          # defaults
    python benchmark.py --docs my_docs/*.pdf      # custom doc path

Output: prints a performance report table to console.
"""
import argparse
import glob
import os
import time
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.document_processor import DocumentProcessor
from src.cache_manager import SemanticCacheManager
from src.vector_store import VectorStoreManager
from src.qa_engine import QAEngine
from src.utils import get_embeddings


def heading(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def bench_document_processing(doc_paths):
    """Time document loading, cleaning, and chunking."""
    heading("1. Document Processing")
    processor = DocumentProcessor()
    results = []

    for path in doc_paths:
        base = os.path.basename(path)
        try:
            t0 = time.perf_counter()
            text = processor.read_document(path)
            read_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            chunks = processor.split_text(text)
            split_time = time.perf_counter() - t1

            total_time = time.perf_counter() - t0
            throughput = len(text) / total_time if total_time > 0 else 0

            results.append({
                "file": base,
                "type": path.split(".")[-1].upper(),
                "chars": len(text),
                "chunks": len(chunks),
                "time_s": round(total_time, 3),
                "chars_per_sec": int(throughput),
                "read_time": round(read_time, 3),
                "split_time": round(split_time, 3),
            })
        except Exception as e:
            results.append({"file": base, "error": str(e)})

    # Print table
    print(f"{'File':<25} {'Type':<6} {'Chars':<8} {'Chunks':<8} "
          f"{'Time(s)':<9} {'Chars/s':<10}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['file']:<25} ERROR: {r['error']}")
        else:
            print(f"{r['file']:<25} {r['type']:<6} {r['chars']:<8} "
                  f"{r['chunks']:<8} {r['time_s']:<9} {r['chars_per_sec']:<10}")
    return results


def bench_vector_store(docs_text):
    """Time embedding + indexing + search."""
    heading("2. Vector Store")
    persist_dir = "./chroma_bench_temp"
    try:
        vs = VectorStoreManager(persist_directory=persist_dir)
        vs.clear_store()

        if docs_text:
            from langchain_core.documents import Document as LCDoc
            avg_chunk_size = 800
            n_docs = max(10, min(50, sum(len(t) for t in docs_text) // avg_chunk_size))
            fake_docs = [LCDoc(page_content=f"Benchmark document {i}. " * 100)
                         for i in range(n_docs)]

            t0 = time.perf_counter()
            result = vs.add_documents(fake_docs, incremental=False)
            add_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            search_results = vs.similarity_search("benchmark", k=4)
            search_time = time.perf_counter() - t1

            print(f"  Documents added : {result['added']}")
            print(f"  Index time      : {add_time:.3f}s")
            print(f"  Search (top-4)  : {search_time*1000:.1f}ms")
            print(f"  Store count     : {vs.get_document_count()}")
        else:
            print("  (skip: no sample docs)")
    finally:
        vs.clear_store()
        import shutil
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir, ignore_errors=True)


def bench_cache():
    """Time cache operations + measure hit rate."""
    heading("3. Cache Performance")
    cache = SemanticCacheManager(cache_dir="cache")
    cache.clear_all()

    # Write throughput
    n = 200
    t0 = time.perf_counter()
    for i in range(n):
        cache.cache_qa(f"question_{i}", "ctx_hash", f"answer_{i}")
    write_time = time.perf_counter() - t0

    # Read throughput
    t0 = time.perf_counter()
    for i in range(n):
        cache.get_qa(f"question_{i}", "ctx_hash")
    read_time = time.perf_counter() - t0

    # Hit rate (all hits — same keys just written)
    stats = cache.get_cache_stats()
    cache.clear_all()

    print(f"  Write throughput  : {n / write_time:.0f} ops/s ({write_time:.3f}s for {n})")
    print(f"  Read throughput   : {n / read_time:.0f} ops/s ({read_time:.3f}s for {n})")
    print(f"  Hit rate (warm)   : {stats['hit_rate']:.1%}")
    print(f"  Cache size        : {stats['total_size_human']}")
    print(f"  Entries           : {stats['total_entries']}")


def bench_semantic_cache():
    """Measure semantic cache similarity computation."""
    heading("4. Semantic Cache (Similarity)")
    cache = SemanticCacheManager(cache_dir="cache")
    cache.clear_all()

    texts = [
        "什么是RAG技术",
        "RAG技术的原理是什么",
        "向量数据库怎么用",
        "ChromaDB支持哪些查询",
        "如何配置embedding模型",
    ]
    for t in texts:
        cache.cache_qa(t, "ctx", f"answer:{t}")

    # Query similar questions (should trigger similarity check)
    queries = [
        "RAG技术原理是什么",       # close to text[1]
        "RAG技术是做什么的",       # close to text[0]
        "ChromaDB支持什么查询方式",  # close to text[3]
        "embedding模型配置方法",     # close to text[4]
        "完全无关的新问题",
    ]
    n_hits = 0
    t0 = time.perf_counter()
    for q in queries:
        result = cache.get_qa(q, "ctx")
        if result:
            n_hits += 1
    elapsed = time.perf_counter() - t0
    print(f"  Semantic queries  : {len(queries)} 次")
    print(f"  Cache hits        : {n_hits} (semantic fuzzy match)")
    print(f"  Avg lookup        : {elapsed / len(queries) * 1000:.1f}ms")

    stats = cache.get_cache_stats()
    print(f"  Hit rate          : {stats['hit_rate']:.1%}")
    cache.clear_all()


def bench_embeddings():
    """Time a single embedding call (needs API key)."""
    heading("5. Embedding Latency")
    try:
        emb = get_embeddings()
        test_texts = ["测试文本用于评估embedding速度"] * 3
        t0 = time.perf_counter()
        emb.embed_documents(test_texts)
        elapsed = time.perf_counter() - t0
        print(f"  3x embedding      : {elapsed:.3f}s avg")
        print(f"  Per query         : {elapsed / 3 * 1000:.1f}ms")
    except Exception as e:
        print(f"  (skip: {e})")


def find_sample_docs():
    """Auto-discover sample documents."""
    paths = []
    for pattern in ["tests/sample_docs/*", "docs/*", "*.pdf", "*.txt", "*.md"]:
        paths.extend(glob.glob(pattern))
    # Only real files (no dirs)
    paths = [p for p in paths if os.path.isfile(p)]
    return paths


def main():
    parser = argparse.ArgumentParser(description="ai-doc-assistant benchmark")
    parser.add_argument("--docs", nargs="*", default=None,
                        help="Sample document paths (default: auto-discover)")
    args = parser.parse_args()

    print("\n" + "█" * 60)
    print("  ai-doc-assistant Performance Benchmark")
    print("█" * 60)

    # Gather documents
    doc_paths = args.docs if args.docs is not None else find_sample_docs()
    docs_text = []

    if doc_paths:
        heading("Sample Documents")
        for p in doc_paths:
            sz = os.path.getsize(p)
            print(f"  {os.path.basename(p):<30} {sz // 1024} KB")

    # Run benchmarks
    if doc_paths:
        bench_document_processing(doc_paths)
        for r in bench_document_processing(doc_paths):
            if "error" not in r:
                docs_text.append("x" * r["chars"])
    else:
        print(f"\n  [INFO] No sample docs found. Place files in tests/sample_docs/")
        print(f"  [INFO] Document processing and vector store tests will be skipped.")
        bench_document_processing([])

    bench_vector_store(docs_text)
    bench_cache()
    bench_semantic_cache()
    bench_embeddings()

    heading("Done")
    print()


if __name__ == "__main__":
    main()
