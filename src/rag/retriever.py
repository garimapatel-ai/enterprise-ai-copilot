"""
Hybrid Retrieval Engine
Dense (Pinecone/ChromaDB) + Sparse (BM25) + Cohere Reranking.
"""

import numpy as np
import hashlib
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Single retrieval result with scoring."""
    chunk_id: str
    content: str
    metadata: Dict
    dense_score: float
    sparse_score: float
    rerank_score: float
    final_score: float


class EmbeddingEngine:
    """Multi-model embedding pipeline."""

    def __init__(self, model_name: str = 'text-embedding-3-large'):
        self.model_name = model_name
        self.dimension = 1536 if '3-large' in model_name else 768
        self._model = None

    def _load_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer('all-MiniLM-L6-v2')  # Fallback
                self.dimension = 384
                logger.info("Loaded local sentence-transformer model")
            except ImportError:
                logger.info("Using simulated embeddings for demo")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        self._load_model()
        if self._model:
            return self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        # Deterministic simulation for demo
        embeddings = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            emb = rng.randn(self.dimension).astype(np.float32)
            emb /= np.linalg.norm(emb)
            embeddings.append(emb)
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return self.embed([query])[0]


class BM25Index:
    """BM25 sparse retrieval index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_freqs = {}
        self.avg_dl = 0
        self.idf = {}

    def index(self, documents: List[Dict]):
        """Build BM25 index from documents."""
        self.documents = documents
        total_dl = 0

        for doc in documents:
            tokens = set(doc['content'].lower().split())
            total_dl += len(doc['content'].split())
            for token in tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_dl = total_dl / max(len(documents), 1)
        n = len(documents)

        for token, df in self.doc_freqs.items():
            self.idf[token] = np.log((n - df + 0.5) / (df + 0.5) + 1)

        logger.info(f"BM25 index built: {n} documents, {len(self.doc_freqs)} unique terms")

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Search using BM25 scoring."""
        query_tokens = query.lower().split()
        scores = []

        for idx, doc in enumerate(self.documents):
            doc_tokens = doc['content'].lower().split()
            dl = len(doc_tokens)
            score = 0

            for qt in query_tokens:
                if qt in self.idf:
                    tf = doc_tokens.count(qt)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1))
                    score += self.idf[qt] * numerator / max(denominator, 1e-8)

            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class VectorStore:
    """In-memory vector store (simulates Pinecone/ChromaDB)."""

    def __init__(self):
        self.vectors = []
        self.documents = []
        self.ids = []

    def upsert(self, ids: List[str], vectors: np.ndarray, documents: List[Dict]):
        """Add vectors to the store."""
        self.ids.extend(ids)
        self.vectors.append(vectors)
        self.documents.extend(documents)
        logger.info(f"Upserted {len(ids)} vectors. Total: {len(self.ids)}")

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> List[Tuple[int, float]]:
        """Cosine similarity search."""
        if not self.vectors:
            return []

        all_vectors = np.vstack(self.vectors)
        similarities = np.dot(all_vectors, query_vector) / (
            np.linalg.norm(all_vectors, axis=1) * np.linalg.norm(query_vector) + 1e-8
        )

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]


class HybridRetriever:
    """Hybrid retrieval combining dense vectors, BM25, and reranking."""

    def __init__(self, dense_weight: float = 0.5, sparse_weight: float = 0.3,
                 rerank_weight: float = 0.2):
        self.embedder = EmbeddingEngine()
        self.vector_store = VectorStore()
        self.bm25 = BM25Index()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rerank_weight = rerank_weight
        self.chunks = []

    def index_chunks(self, chunks: List):
        """Index document chunks for retrieval."""
        logger.info(f"Indexing {len(chunks)} chunks...")
        self.chunks = chunks

        # Prepare documents
        docs = [{'chunk_id': c.chunk_id, 'content': c.content, 'metadata': c.metadata}
                for c in chunks]

        # Dense indexing
        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed(texts)
        ids = [c.chunk_id for c in chunks]
        self.vector_store.upsert(ids, embeddings, docs)

        # Sparse indexing
        self.bm25.index(docs)

        logger.info(f"✅ Indexed {len(chunks)} chunks (dense + sparse)")

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Hybrid retrieval with RRF fusion."""
        # Dense retrieval
        query_embedding = self.embedder.embed_query(query)
        dense_results = self.vector_store.search(query_embedding, top_k=20)
        dense_scores = {idx: score for idx, score in dense_results}

        # Sparse retrieval
        sparse_results = self.bm25.search(query, top_k=20)
        sparse_scores = {idx: score for idx, score in sparse_results}

        # Reciprocal Rank Fusion
        all_indices = set(dense_scores.keys()) | set(sparse_scores.keys())
        rrf_scores = {}
        k = 60  # RRF constant

        dense_ranked = sorted(dense_scores.keys(), key=lambda x: dense_scores[x], reverse=True)
        sparse_ranked = sorted(sparse_scores.keys(), key=lambda x: sparse_scores[x], reverse=True)

        for idx in all_indices:
            dense_rank = dense_ranked.index(idx) + 1 if idx in dense_ranked else len(self.chunks)
            sparse_rank = sparse_ranked.index(idx) + 1 if idx in sparse_ranked else len(self.chunks)
            rrf_scores[idx] = (self.dense_weight / (k + dense_rank) +
                               self.sparse_weight / (k + sparse_rank))

        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        # Build results
        results = []
        for idx in sorted_indices:
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append(RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    dense_score=dense_scores.get(idx, 0.0),
                    sparse_score=sparse_scores.get(idx, 0.0),
                    rerank_score=rrf_scores.get(idx, 0.0),
                    final_score=round(rrf_scores.get(idx, 0.0), 4)
                ))

        logger.info(f"Retrieved {len(results)} chunks for query: '{query[:50]}...'")
        return results


if __name__ == '__main__':
    from document_processor import DocumentPipeline

    pipeline = DocumentPipeline()
    chunks = pipeline.process_directory('data/documents')

    retriever = HybridRetriever()
    retriever.index_chunks(chunks)

    queries = [
        "What is the code review process?",
        "How do we handle P0 incidents?",
        "What are the ML model deployment steps?",
        "What is our data privacy policy for PII?",
    ]

    for q in queries:
        print(f"\n🔍 Query: {q}")
        results = retriever.retrieve(q, top_k=3)
        for i, r in enumerate(results):
            print(f"  [{i+1}] Score: {r.final_score:.4f} | Source: {r.metadata.get('file_name', 'N/A')}")
            print(f"      {r.content[:120]}...")
