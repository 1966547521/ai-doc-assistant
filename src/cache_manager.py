"""
Semantic Cache Manager for LLM Responses
Implements both text similarity based semantic caching and simple hash-based caching
"""

import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    key: str
    value: Any
    timestamp: float
    ttl: Optional[float] = None
    access_count: int = 0
    last_access: float = 0.0


class SemanticCacheManager:
    """
    Advanced cache manager combining semantic similarity and LRU eviction
    """

    def __init__(
        self,
        cache_dir: str = "cache",
        max_entries: int = 1000,
        default_ttl: float = 3600 * 24,  # 24 hours
        similarity_threshold: float = 0.25,  # Jaccard similarity threshold for semantic match
    ):
        self.cache_dir = cache_dir
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.similarity_threshold = similarity_threshold

        # LRU cache - OrderedDict keeps insertion order
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._load_cache()

        # Hit/miss tracking for accurate hit rate
        self._hits = 0
        self._misses = 0

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

    def _hash_key(self, key: str) -> str:
        """Generate SHA-256 hash for cache key"""
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        """Load cache from disk"""
        cache_file = os.path.join(self.cache_dir, "semantic_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for entry_data in data:
                        entry = CacheEntry(**entry_data)
                        self._cache[entry.key] = entry
            except (IOError, json.JSONDecodeError) as e:
                logger.warning("Failed to load cache: %s", e)
                self._cache = OrderedDict()

    def _save_cache(self) -> None:
        """Save cache to disk"""
        cache_file = os.path.join(self.cache_dir, "semantic_cache.json")
        try:
            data = [asdict(entry) for entry in self._cache.values()]
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.warning("Failed to save cache: %s", e)

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        if entry.ttl is None:
            return False
        return time.time() - entry.timestamp > entry.ttl

    def _evict_if_needed(self) -> None:
        """Evict least recently used entries if cache size exceeds limit"""
        # First remove expired entries
        expired_keys = [k for k, v in self._cache.items() if self._is_expired(v)]
        for k in expired_keys:
            del self._cache[k]

        # Then remove LRU entries if still over limit
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Set a value in cache with optional TTL

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None = use default)
        """
        hashed_key = self._hash_key(key)
        ttl = ttl if ttl is not None else self.default_ttl
        current_time = time.time()

        entry = CacheEntry(
            key=hashed_key,
            value=value,
            timestamp=current_time,
            ttl=ttl,
            access_count=0,
            last_access=current_time,
        )

        # Move to end (most recently used)
        if hashed_key in self._cache:
            del self._cache[hashed_key]
        self._cache[hashed_key] = entry

        self._evict_if_needed()
        self._save_cache()

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Get a value from cache

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        hashed_key = self._hash_key(key)

        if hashed_key not in self._cache:
            self._misses += 1
            return default

        entry = self._cache[hashed_key]

        if self._is_expired(entry):
            del self._cache[hashed_key]
            self._save_cache()
            self._misses += 1
            return default

        self._hits += 1
        # Update access stats and move to end (MRU)
        entry.access_count += 1
        entry.last_access = time.time()
        del self._cache[hashed_key]
        self._cache[hashed_key] = entry

        self._save_cache()
        return entry.value

    def get_with_similarity(
        self,
        text: str,
        similar_texts: List[str],
    ) -> Optional[Tuple[str, Any]]:
        """
        Get cached value by semantic similarity

        Args:
            text: Text to match
            similar_texts: List of candidate texts to compare

        Returns:
            Tuple of (matched_text, cached_value) or None
        """
        for candidate in similar_texts:
            if self._compute_similarity(text, candidate) >= self.similarity_threshold:
                value = self.get(candidate, None)
                if value is not None:
                    return (candidate, value)
        return None

    @staticmethod
    def _compute_similarity(text1: str, text2: str) -> float:
        """
        Character n-gram Jaccard similarity with adaptive n based on text length.
        Short text (e.g., Chinese questions) uses bigrams for better overlap.
        """
        avg_len = (len(text1) + len(text2)) / 2
        n = 2 if avg_len < 50 else 3

        def get_ngrams(s: str) -> set:
            s = s.lower()
            return set(s[i : i + n] for i in range(len(s) - n + 1))

        ngrams1 = get_ngrams(text1)
        ngrams2 = get_ngrams(text2)

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        return intersection / union if union > 0 else 0.0

    def clear(self) -> None:
        """Clear all cache entries"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        self._save_cache()

    def cache_document_summary(self, document_text: str, summary: str) -> None:
        """Cache document summaries"""
        key = f"summary:{self._hash_key(document_text[:5000])}"
        self.set(key, summary)

    def get_document_summary(self, document_text: str) -> Optional[str]:
        """Get cached document summary"""
        key = f"summary:{self._hash_key(document_text[:5000])}"
        return self.get(key, None)

    def cache_qa(self, question: str, context_hash: str, answer: str) -> None:
        """Cache QA response with question text for semantic lookup."""
        question_hash = self._hash_key(question)
        key = f"qa:{context_hash}:{question_hash}"
        self.set(key, {"q": question, "a": answer}, ttl=3600)

    def get_qa(self, question: str, context_hash: str) -> Optional[str]:
        """Get cached QA response by exact hash match."""
        question_hash = self._hash_key(question)
        key = f"qa:{context_hash}:{question_hash}"
        value = self.get(key, None)
        if isinstance(value, dict):
            return value.get("a")
        if isinstance(value, str):
            return value
        return None

    def get_qa_semantic(
        self, question: str, context_hash: str, threshold: Optional[float] = None
    ) -> Optional[str]:
        """Get cached QA response with semantic similarity fallback.

        First tries exact match, then scans all cached QA entries
        for similar questions using Jaccard trigram similarity.

        Args:
            question: The question to look up
            context_hash: Document context hash for exact match
            threshold: Override similarity_threshold

        Returns:
            Cached answer or None
        """
        # 1. Exact match first (fast path)
        exact = self.get_qa(question, context_hash)
        if exact is not None:
            return exact

        # 2. Scan cached QA entries for semantic similarity
        thresh = threshold if threshold is not None else self.similarity_threshold
        best_score = 0.0
        best_answer = None

        for entry in list(self._cache.values()):
            value = entry.value
            if not isinstance(value, dict):
                continue
            cached_q = value.get("q")
            if not cached_q:
                continue
            score = self._compute_similarity(question, cached_q)
            if score >= thresh and score > best_score:
                best_score = score
                best_answer = value.get("a")

        if best_answer is not None:
            self._hits += 1
        return best_answer

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = len(self._cache)
        total_size = 0

        for entry in self._cache.values():
            try:
                total_size += len(json.dumps(entry.value, ensure_ascii=False))
            except (TypeError, ValueError):
                pass

        return {
            "total_entries": total_entries,
            "total_size": total_size,
            "total_size_human": self._format_size(total_size),
            "hit_rate": self._calculate_hit_rate(),
        }

    def clear_all(self) -> None:
        """Clear all cache including old files"""
        self.clear()
        # Clear old cache files for backward compatibility
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".json") and filename != "semantic_cache.json":
                    try:
                        os.remove(os.path.join(self.cache_dir, filename))
                    except OSError:
                        pass

    @staticmethod
    def _format_size(size_bytes: float) -> str:
        """Format bytes to human readable format"""
        if size_bytes == 0:
            return "0 B"
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"

    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate based on tracked hits and misses."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return round(self._hits / total, 4)


# For backward compatibility
CacheManager = SemanticCacheManager
