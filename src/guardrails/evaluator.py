"""
RAGAS-style RAG Evaluator — measures faithfulness, context relevancy, and answer completeness.
"""

import numpy as np
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGEvalResult:
    question: str
    faithfulness: float          # Is the answer supported by context? [0,1]
    context_relevancy: float     # Is retrieved context relevant to question? [0,1]
    answer_relevancy: float      # Does answer address the question? [0,1]
    context_recall: float        # Does context cover all needed info? [0,1]
    overall_score: float
    grade: str                   # A/B/C/D/F


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality using RAGAS-inspired metrics.
    Simulates LLM-based evaluation — replace scoring methods with actual NLI/LLM calls in prod.
    """

    def __init__(self):
        self.results: List[RAGEvalResult] = []
        logger.info("RAG Evaluator initialized")

    def evaluate(self, question: str, answer: str,
                 contexts: List[str], ground_truth: Optional[str] = None) -> RAGEvalResult:
        """Evaluate a single RAG response."""
        faithfulness = self._score_faithfulness(answer, contexts)
        ctx_relevancy = self._score_context_relevancy(question, contexts)
        ans_relevancy = self._score_answer_relevancy(question, answer)
        ctx_recall = self._score_context_recall(answer, ground_truth, contexts) if ground_truth else ctx_relevancy

        overall = np.mean([faithfulness, ctx_relevancy, ans_relevancy, ctx_recall])
        grade = self._grade(overall)

        result = RAGEvalResult(
            question=question, faithfulness=faithfulness,
            context_relevancy=ctx_relevancy, answer_relevancy=ans_relevancy,
            context_recall=ctx_recall, overall_score=round(overall, 3), grade=grade
        )
        self.results.append(result)
        logger.info(f"RAG eval — overall: {overall:.3f} ({grade})")
        return result

    def batch_evaluate(self, samples: List[Dict]) -> List[RAGEvalResult]:
        """Evaluate a batch of RAG samples."""
        results = []
        for s in samples:
            r = self.evaluate(
                s['question'], s['answer'], s['contexts'],
                s.get('ground_truth')
            )
            results.append(r)
        logger.info(f"Batch evaluation complete: {len(results)} samples")
        return results

    def _score_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Check if claims in answer are supported by context."""
        if not contexts:
            return 0.5
        ans_tokens = set(answer.lower().split())
        ctx_tokens = set(' '.join(contexts).lower().split())
        overlap = len(ans_tokens & ctx_tokens)
        return round(min(1.0, overlap / (len(ans_tokens) + 1e-8) * 1.5), 3)

    def _score_context_relevancy(self, question: str, contexts: List[str]) -> float:
        """Check if retrieved contexts are relevant to the question."""
        if not contexts:
            return 0.0
        q_tokens = set(question.lower().split())
        scores = []
        for ctx in contexts:
            ctx_tokens = set(ctx.lower().split())
            overlap = len(q_tokens & ctx_tokens)
            scores.append(overlap / (len(q_tokens) + 1e-8))
        return round(min(1.0, np.mean(scores) * 2.0), 3)

    def _score_answer_relevancy(self, question: str, answer: str) -> float:
        """Check if answer is relevant to the question."""
        q_tokens = set(question.lower().split())
        a_tokens = set(answer.lower().split())
        overlap = len(q_tokens & a_tokens)
        length_factor = min(1.0, len(a_tokens) / 20)
        return round(min(1.0, (overlap / (len(q_tokens) + 1e-8)) * 1.5 + length_factor * 0.2), 3)

    def _score_context_recall(self, answer: str, ground_truth: str, contexts: List[str]) -> float:
        """Check if ground truth is covered by retrieved context."""
        gt_tokens = set(ground_truth.lower().split())
        ctx_tokens = set(' '.join(contexts).lower().split())
        overlap = len(gt_tokens & ctx_tokens)
        return round(min(1.0, overlap / (len(gt_tokens) + 1e-8)), 3)

    def _grade(self, score: float) -> str:
        if score >= 0.9: return 'A'
        if score >= 0.8: return 'B'
        if score >= 0.7: return 'C'
        if score >= 0.6: return 'D'
        return 'F'

    def summary(self) -> Dict:
        if not self.results:
            return {'samples': 0}
        grades = [r.grade for r in self.results]
        return {
            'samples_evaluated': len(self.results),
            'avg_faithfulness': round(np.mean([r.faithfulness for r in self.results]), 3),
            'avg_context_relevancy': round(np.mean([r.context_relevancy for r in self.results]), 3),
            'avg_answer_relevancy': round(np.mean([r.answer_relevancy for r in self.results]), 3),
            'avg_overall': round(np.mean([r.overall_score for r in self.results]), 3),
            'grade_distribution': {g: grades.count(g) for g in 'ABCDF'},
        }


if __name__ == '__main__':
    evaluator = RAGEvaluator()

    samples = [
        {
            'question': 'What is the RAG factual accuracy of the AI copilot?',
            'answer': 'The AI copilot achieves 96.2% factual accuracy as measured by RAGAS evaluation.',
            'contexts': ['The AI copilot achieved 96.2% factual accuracy in internal benchmarks using RAGAS framework.'],
            'ground_truth': '96.2% factual accuracy'
        },
        {
            'question': 'How many documents are indexed in the knowledge base?',
            'answer': 'Over 500,000 enterprise documents are indexed from Confluence, SharePoint, and email.',
            'contexts': ['The system indexes 500K+ internal documents from Confluence, Slack, and Jira.'],
            'ground_truth': '500,000+ documents'
        },
        {
            'question': 'What is the average response latency?',
            'answer': 'The system responds in under 2 seconds for most queries.',
            'contexts': ['Average response latency is 1.8 seconds across production workloads.'],
            'ground_truth': '1.8 seconds average latency'
        },
    ]

    results = evaluator.batch_evaluate(samples)
    for r in results:
        print(f"\nQ: {r.question[:60]}")
        print(f"  Faithfulness: {r.faithfulness} | Context: {r.context_relevancy} | Answer: {r.answer_relevancy}")
        print(f"  Overall: {r.overall_score} ({r.grade})")

    print(f"\nSummary: {evaluator.summary()}")
