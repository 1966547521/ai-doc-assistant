"""Vector store management with incremental updates and deduplication."""

import hashlib
import os
import shutil
import logging
from typing import Dict, List, Optional, Set

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.utils import get_embeddings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages the vector database with incremental updates and deduplication."""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self._embeddings = None
        self.vector_store: Optional[Chroma] = None
        self._document_ids: Set[str] = set()
        self._load_document_ids()

    @property
    def embeddings(self):
        """Lazy-load embeddings to avoid connection errors at import time."""
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    def _compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of document content for deduplication."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _load_document_ids(self) -> None:
        """Load existing document IDs from vector store."""
        try:
            if self.vector_store is None:
                self.init_store()
            if self.vector_store:
                # Get all document hashes from metadata
                all_ids = self.vector_store.get(include=["metadatas"])
                if all_ids and "metadatas" in all_ids:
                    for meta in all_ids["metadatas"]:
                        if meta and "content_hash" in meta:
                            self._document_ids.add(meta["content_hash"])
        except (IOError, ValueError) as e:
            logger.warning("Failed to load document IDs: %s", e)

    def init_store(self) -> None:
        """Initialize or load the vector store."""
        self.vector_store = Chroma(
            embedding_function=self.embeddings, persist_directory=self.persist_directory
        )
        self._load_document_ids()

    def _get_collection_dimension(self) -> Optional[int]:
        """Get the dimension of existing collection from stored embeddings."""
        try:
            if os.path.exists(self.persist_directory):
                import chromadb

                client = chromadb.PersistentClient(path=self.persist_directory)
                collections = client.list_collections()
                if collections:
                    sample = collections[0].get(limit=1, include=["embeddings"])
                    if sample:
                        embeddings = sample.get("embeddings")
                        if embeddings is not None and len(embeddings) > 0:
                            return len(embeddings[0])  # type: ignore
            return None
        except (IOError, ImportError):
            return None

    def _check_dimension(self) -> bool:
        """Check if current embedding dimension matches existing collection."""
        existing_dim = self._get_collection_dimension()
        if existing_dim is None:
            return True
        test_embedding = self.embeddings.embed_query("test")
        current_dim = len(test_embedding)
        if existing_dim != current_dim:
            logger.warning(
                "Dimension mismatch: collection=%s, current=%s", existing_dim, current_dim
            )
            return False
        return True

    def add_documents(
        self, documents: List[Document], incremental: bool = True
    ) -> Dict[str, int]:
        """
        Add documents to the vector store with incremental updates and deduplication.

        Args:
            documents: Documents to add
            incremental: If True, perform incremental update (skip duplicates)

        Returns:
            Dict with stats: {'added': int, 'skipped': int, 'total': int}
        """
        if not self.vector_store:
            self.init_store()

        if not self._check_dimension():
            logger.warning("Dimension mismatch detected! Recreating vector store...")
            self._recreate_store()
            if not self.vector_store:
                self.init_store()

        added = 0
        skipped = 0

        # Filter out duplicates
        filtered_docs: List[Document] = []
        for doc in documents:
            content_hash = self._compute_content_hash(doc.page_content)

            if incremental and content_hash in self._document_ids:
                skipped += 1
                continue

            # Store content hash in metadata for future deduplication
            doc.metadata["content_hash"] = content_hash
            filtered_docs.append(doc)

        if filtered_docs:
            try:
                if self.vector_store:
                    self.vector_store.add_documents(filtered_docs)
                    added = len(filtered_docs)

                    # Update our tracked document IDs
                    for doc in filtered_docs:
                        self._document_ids.add(doc.metadata["content_hash"])

            except (RuntimeError, ValueError) as e:
                if "dimension" in str(e).lower():
                    logger.warning("Dimension error detected (%s), recreating store...", e)
                    self._recreate_store()
                    self.init_store()
                    if self.vector_store:
                        self.vector_store.add_documents(filtered_docs)
                        added = len(filtered_docs)
                        for doc in filtered_docs:
                            self._document_ids.add(doc.metadata["content_hash"])
                else:
                    raise

        return {"added": added, "skipped": skipped, "total": len(documents)}

    def _recreate_store(self) -> None:
        """Delete and recreate the vector store."""
        self.clear_store()
        if os.path.exists(self.persist_directory):
            try:
                import chromadb

                client = chromadb.PersistentClient(path=self.persist_directory)
                for col in client.list_collections():
                    try:
                        client.delete_collection(col.name)
                    except RuntimeError:
                        pass
            except (ImportError, RuntimeError):
                pass
            shutil.rmtree(self.persist_directory, ignore_errors=True)
        self.vector_store = None
        self._document_ids.clear()

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents."""
        if not self.vector_store:
            self.init_store()
        if self.vector_store:
            return self.vector_store.similarity_search(query, k=k)
        return []

    def clear_store(self) -> None:
        """Clear all documents from the vector store."""
        if self.vector_store:
            try:
                self.vector_store.delete_collection()
            except RuntimeError:
                pass
            self.vector_store = None
            self._document_ids.clear()

    def get_store(self) -> Optional[Chroma]:
        """Get the vector store instance."""
        if not self.vector_store:
            self.init_store()
        return self.vector_store

    def get_document_count(self) -> int:
        """Get the number of documents in the vector store."""
        if not self.vector_store:
            return 0
        try:
            result = self.vector_store.get()
            return len(result.get("ids", [])) if result else 0
        except (RuntimeError, ValueError):
            return 0

    @staticmethod
    def create_with_fallback(persist_directory: str = "./chroma_db"):
        """Create VectorStoreManager with fallback to local embeddings."""
        try:
            return VectorStoreManager(persist_directory)
        except (RuntimeError, ImportError) as e:
            logger.warning("Failed to create vector store: %s", e)
            return VectorStoreManager(persist_directory)
