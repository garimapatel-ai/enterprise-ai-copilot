"""
Dual Embedding Pipeline — dense (text-embedding-3-large) + sparse (BM25) for hybrid retrieval.
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddedChunk:
    chunk_id: str
    content: str
    dense_vector: np.ndarray
    sparse_vector: Dict[str, float]
    metadata: Dict


class DenseEmbedder:
    """
    Simulates text-embedding-3-large (OpenAI) or bge-large-en (HuggingFace).
    In production: replace _embed() with actual API/model call.
    """

    EMBEDDING_DIM = 1536   # text-embedding-3-large dimension

    def __init__(self, model: str = 'text-embedding-3-large'):
        self.model = model
        logger.info(f"Dense embedder initialized: {model} (dim={self.EMBEDDING_DIM})")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts. Returns (N, dim) array."""
        embeddings = np.array([self._embed(t) for t in texts])
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-10)

    def _embed(self, text: str) -> np.ndarray:
        # Deterministic simulation based on text hash
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        return rng.randn(self.EMBEDDING_DIM).astype(np.float32)


class BM25Embedder:
    """
    BM25 sparse embedding — produces term-weight dictionaries for keyword matching.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.df: Counter = Counter()
        self.corpus_size = 0
        self.avg_dl = 0.0
        self._fitted = False

    def fit(self, texts: List[str]):
        """Fit BM25 on a corpus to compute IDF weights."""
        self.corpus_size = len(texts)
        all_dls = []
        for text in texts:
            tokens = set(self._tokenize(text))
            self.df.update(tokens)
            all_dls.append(len(self._tokenize(text)))
        self.avg_dl = np.mean(all_dls) if all_dls else 1.0
        self._fitted = True
        logger.info(f"BM25 fitted on {self.corpus_size} docs, vocab size: {len(self.df)}")

    def embed(self, text: str) -> Dict[str, float]:
        """Compute BM25 sparse vector for a single text."""
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        dl = len(tokens)
        sparse = {}
        for term, freq in tf.items():
            if not self._fitted or term not in self.df:
                idf = 1.0
            else:
                idf = math.log((self.corpus_size - self.df[term] + 0.5) /
                               (self.df[term] + 0.5) + 1)
            tf_norm = (freq * (self.k1 + 1)) / (
                freq + self.k1 * (1 - self.b + self.b * dl / (self.avg_dl + 1e-8))
            )
            sparse[term] = round(idf * tf_norm, 4)
        return sparse

    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())


class HybridEmbeddingPipeline:
    """Combines dense and sparse embeddings for robust hybrid retrieval."""

    def __init__(self, dense_model: str = 'text-embedding-3-large',
                 batch_size: int = 64):
        self.dense = DenseEmbedder(dense_model)
        self.sparse = BM25Embedder()
        self.batch_size = batch_size
        self.embedded_chunks: List[EmbeddedChunk] = []

    def fit_and_embed(self, chunks) -> List[EmbeddedChunk]:
        """Fit BM25 on corpus and embed all chunks."""
        texts = [c.content for c in chunks]
        logger.info(f"Embedding {len(chunks)} chunks...")

        # Fit BM25 on full corpus
        self.sparse.fit(texts)

        embedded = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_texts = [c.content for c in batch]

            dense_vecs = self.dense.embed(batch_texts)
            for j, chunk in enumerate(batch):
                embedded.append(EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    dense_vector=dense_vecs[j],
                    sparse_vector=self.sparse.embed(chunk.content),
                    metadata=chunk.metadata
                ))
            logger.debug(f"Embedded batch {i//self.batch_size + 1}")

        self.embedded_chunks = embedded
        logger.info(f"Embedding complete: {len(embedded)} chunks")
        return embedded

    def embed_query(self, query: str) -> Tuple[np.ndarray, Dict[str, float]]:
        """Embed a query for retrieval."""
        dense_vec = self.dense.embed([query])[0]
        sparse_vec = self.sparse.embed(query)
        return dense_vec, sparse_vec


if __name__ == '__main__':
    from src.rag.chunker import SemanticChunker

    chunker = SemanticChunker(max_tokens=256, overlap_tokens=32)
    docs = [
        {'doc_id': 'doc1', 'content': 'Enterprise AI systems require robust RAG pipelines with hybrid retrieval for accurate knowledge access.', 'metadata': {}},
        {'doc_id': 'doc2', 'content': 'Multi-agent orchestration enables specialized AI agents to collaborate on complex enterprise tasks.', 'metadata': {}},
        {'doc_id': 'doc3', 'content': 'Fine-tuning LLMs with QLoRA reduces GPU memory requirements while maintaining model quality.', 'metadata': {}},
    ]
    chunks = chunker.batch_chunk(docs)

    pipeline = HybridEmbeddingPipeline()
    embedded = pipeline.fit_and_embed(chunks)

    print(f"\nEmbedded {len(embedded)} chunks")
    print(f"Dense vector dim: {embedded[0].dense_vector.shape}")
    print(f"Sparse terms: {list(embedded[0].sparse_vector.keys())[:5]}")

    q_dense, q_sparse = pipeline.embed_query("How do RAG systems work?")
    print(f"\nQuery dense dim: {q_dense.shape}")
    print(f"Query sparse terms: {list(q_sparse.keys())[:5]}")
