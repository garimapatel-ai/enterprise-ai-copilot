"""Tests for RAG pipeline components."""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rag.chunker import SemanticChunker
from src.rag.embedder import DenseEmbedder, BM25Embedder, HybridEmbeddingPipeline
from src.rag.reranker import CrossEncoderReranker, ReciprocaRankFusion
from src.guardrails.hallucination_detector import HallucinationDetector
from src.guardrails.evaluator import RAGEvaluator


class TestChunker:
    def setup_method(self):
        self.chunker = SemanticChunker(max_tokens=128, overlap_tokens=16)

    def test_chunks_non_empty(self):
        chunks = self.chunker.chunk("Hello world. " * 100, 'doc1')
        assert len(chunks) > 0

    def test_chunk_ids_unique(self):
        chunks = self.chunker.chunk("Hello world. " * 100, 'doc1')
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_doc_id_preserved(self):
        chunks = self.chunker.chunk("Some text here.", 'my_doc')
        assert all(c.doc_id == 'my_doc' for c in chunks)

    def test_batch_chunk(self):
        docs = [{'doc_id': f'd{i}', 'content': f'Document {i} content. ' * 20} for i in range(5)]
        chunks = self.chunker.batch_chunk(docs)
        assert len(chunks) >= 5


class TestEmbedder:
    def test_dense_embed_shape(self):
        embedder = DenseEmbedder()
        vecs = embedder.embed(['Hello world', 'Enterprise AI'])
        assert vecs.shape == (2, 1536)

    def test_dense_embed_normalized(self):
        import numpy as np
        embedder = DenseEmbedder()
        vecs = embedder.embed(['test'])
        norms = np.linalg.norm(vecs, axis=1)
        assert abs(norms[0] - 1.0) < 0.01

    def test_bm25_fit_and_embed(self):
        bm25 = BM25Embedder()
        corpus = ['RAG systems use retrieval', 'LLMs generate text', 'Agents orchestrate tasks']
        bm25.fit(corpus)
        vec = bm25.embed('RAG retrieval')
        assert len(vec) > 0
        assert all(v >= 0 for v in vec.values())

    def test_pipeline_produces_embedded_chunks(self):
        from src.rag.chunker import SemanticChunker
        chunker = SemanticChunker(max_tokens=128)
        docs = [{'doc_id': f'd{i}', 'content': f'Enterprise knowledge document {i}. ' * 10} for i in range(3)]
        chunks = chunker.batch_chunk(docs)
        pipeline = HybridEmbeddingPipeline()
        embedded = pipeline.fit_and_embed(chunks)
        assert len(embedded) == len(chunks)
        assert embedded[0].dense_vector is not None


class TestReranker:
    def setup_method(self):
        self.reranker = CrossEncoderReranker(top_n=3)
        self.candidates = [
            {'chunk_id': f'c{i}', 'content': f'Document about RAG topic {i}.', 'score': 0.9 - i*0.1, 'metadata': {}}
            for i in range(5)
        ]

    def test_returns_top_n(self):
        results = self.reranker.rerank("RAG retrieval", self.candidates, top_n=3)
        assert len(results) == 3

    def test_final_ranks_assigned(self):
        results = self.reranker.rerank("query", self.candidates)
        assert results[0].final_rank == 1

    def test_rrf_fuses_results(self):
        rrf = ReciprocaRankFusion()
        fused = rrf.fuse(self.candidates[:3], self.candidates[2:])
        assert len(fused) <= 5
        assert all('score' in d for d in fused)


class TestHallucinationDetector:
    def setup_method(self):
        self.detector = HallucinationDetector(entailment_threshold=0.3)
        self.sources = ['The system achieves 96% accuracy on benchmark tests.',
                        'Average latency is 1.8 seconds per query.']

    def test_grounded_response(self):
        resp = "The system achieves 96% accuracy with 1.8 second latency."
        report = self.detector.check(resp, self.sources)
        assert report.risk_level in ('low', 'medium')

    def test_no_claims_is_safe(self):
        resp = "The system works well and provides helpful responses."
        report = self.detector.check(resp, self.sources)
        assert report.is_grounded

    def test_stats_tracks_checks(self):
        self.detector.check("Test 99.9% accuracy.", self.sources)
        self.detector.check("Latency 1.8 seconds.", self.sources)
        stats = self.detector.stats()
        assert stats['total_checks'] == 2
