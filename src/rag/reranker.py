"""
Reranker — Cohere-style cross-encoder reranking for final relevance scoring.
Takes top-k candidates from hybrid retrieval and reorders by semantic relevance.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    chunk_id: str
    content: str
    initial_rank: int
    final_rank: int
    rerank_score: float
    initial_score: float
    metadata: Dict


class CrossEncoderReranker:
    """
    Cross-encoder reranker that jointly encodes query + document pairs.
    Simulates Cohere Rerank API behavior.
    In production: replace _score_pair() with actual Cohere API call.
    """

    def __init__(self, model: str = 'rerank-english-v3.0', top_n: int = 5):
        self.model = model
        self.top_n = top_n
        logger.info(f"Reranker initialized: {model}, top_n={top_n}")

    def rerank(self, query: str, candidates: List[Dict],
               top_n: Optional[int] = None) -> List[RankedResult]:
        """
        Rerank candidate chunks by joint query-document relevance.
        candidates: list of dicts with 'chunk_id', 'content', 'score', 'metadata'
        """
        top_n = top_n or self.top_n
        logger.info(f"Reranking {len(candidates)} candidates for: {query[:60]}")

        scored = []
        for i, cand in enumerate(candidates):
            rerank_score = self._score_pair(query, cand['content'])
            scored.append(RankedResult(
                chunk_id=cand['chunk_id'],
                content=cand['content'],
                initial_rank=i + 1,
                final_rank=0,
                rerank_score=rerank_score,
                initial_score=cand.get('score', 0.0),
                metadata=cand.get('metadata', {})
            ))

        scored.sort(key=lambda x: x.rerank_score, reverse=True)
        for i, r in enumerate(scored):
            r.final_rank = i + 1

        top_results = scored[:top_n]
        rank_changes = sum(1 for r in top_results if r.initial_rank != r.final_rank)
        logger.info(f"Reranking complete — {rank_changes}/{len(top_results)} positions changed")

        return top_results

    def _score_pair(self, query: str, document: str) -> float:
        """
        Compute relevance score for a query-document pair.
        Simulates cross-encoder scoring via keyword overlap + length penalty.
        """
        q_tokens = set(query.lower().split())
        d_tokens = set(document.lower().split())

        # Jaccard-based overlap
        overlap = len(q_tokens & d_tokens)
        union = len(q_tokens | d_tokens)
        jaccard = overlap / (union + 1e-8)

        # Prefer docs with query terms in the beginning
        doc_start = document[:200].lower()
        position_bonus = sum(0.02 for t in q_tokens if t in doc_start)

        # Mild length normalization (very short or very long docs penalized)
        length_factor = min(1.0, len(document.split()) / 50)

        score = jaccard * 0.6 + position_bonus + length_factor * 0.1
        # Add small noise for realism
        seed = hash(query + document) % (2**31)
        noise = np.random.RandomState(seed).uniform(-0.02, 0.02)
        return round(float(np.clip(score + noise, 0.0, 1.0)), 4)


class ReciprocaRankFusion:
    """
    RRF — combines rankings from dense and sparse retrieval before reranking.
    score(d) = Σ 1 / (k + rank_i(d))
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, dense_results: List[Dict], sparse_results: List[Dict]) -> List[Dict]:
        """Fuse two ranked lists using Reciprocal Rank Fusion."""
        scores: Dict[str, float] = {}
        all_docs: Dict[str, Dict] = {}

        for rank, doc in enumerate(dense_results, 1):
            cid = doc['chunk_id']
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.k + rank)
            all_docs[cid] = doc

        for rank, doc in enumerate(sparse_results, 1):
            cid = doc['chunk_id']
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.k + rank)
            all_docs[cid] = doc

        fused = sorted(all_docs.values(), key=lambda d: scores[d['chunk_id']], reverse=True)
        for doc in fused:
            doc['score'] = round(scores[doc['chunk_id']], 5)

        logger.debug(f"RRF fused {len(dense_results)}+{len(sparse_results)} → {len(fused)} unique results")
        return fused


from typing import Optional


if __name__ == '__main__':
    rrf = ReciprocaRankFusion(k=60)
    reranker = CrossEncoderReranker(top_n=3)

    dense = [
        {'chunk_id': 'c1', 'content': 'RAG systems combine retrieval and generation for accurate AI responses.', 'score': 0.92, 'metadata': {}},
        {'chunk_id': 'c2', 'content': 'Pinecone vector database enables fast similarity search at scale.', 'score': 0.85, 'metadata': {}},
        {'chunk_id': 'c3', 'content': 'Hybrid retrieval improves RAG factual accuracy by combining dense and sparse search.', 'score': 0.81, 'metadata': {}},
    ]
    sparse = [
        {'chunk_id': 'c3', 'content': 'Hybrid retrieval improves RAG factual accuracy by combining dense and sparse search.', 'score': 0.78, 'metadata': {}},
        {'chunk_id': 'c4', 'content': 'BM25 keyword search captures exact term matches that dense embeddings may miss.', 'score': 0.74, 'metadata': {}},
        {'chunk_id': 'c1', 'content': 'RAG systems combine retrieval and generation for accurate AI responses.', 'score': 0.70, 'metadata': {}},
    ]

    fused = rrf.fuse(dense, sparse)
    query = "How does RAG improve AI accuracy?"
    final = reranker.rerank(query, fused)

    print(f"Top {len(final)} results after reranking:")
    for r in final:
        print(f"  [{r.final_rank}] (was #{r.initial_rank}) score={r.rerank_score:.4f}: {r.content[:70]}")
