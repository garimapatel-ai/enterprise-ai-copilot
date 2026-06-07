"""
AI Safety Guardrails & Hallucination Detection
NeMo-style content safety + RAGAS evaluation framework.
"""

import numpy as np
import re
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SafetyCheckResult:
    """Result of a safety check."""
    is_safe: bool
    category: str
    confidence: float
    reason: str


@dataclass
class RAGASMetrics:
    """RAGAS evaluation metrics."""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall_score: float


class ContentSafetyGuard:
    """Content safety checker inspired by NeMo Guardrails."""

    BLOCKED_PATTERNS = [
        r'(?i)(hack|exploit|bypass|crack)\s+(password|security|system)',
        r'(?i)(create|make|build)\s+(malware|virus|trojan)',
        r'(?i)(how to|ways to)\s+(steal|scam|fraud)',
    ]

    SENSITIVE_TOPICS = [
        'personal_info', 'financial_data', 'medical_records',
        'trade_secrets', 'classified', 'legal_advice'
    ]

    PII_PATTERNS = {
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
    }

    def check_input(self, text: str) -> SafetyCheckResult:
        """Check user input for safety violations."""
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, text):
                return SafetyCheckResult(
                    is_safe=False,
                    category='harmful_intent',
                    confidence=0.95,
                    reason=f'Input matches blocked pattern'
                )

        return SafetyCheckResult(
            is_safe=True, category='clean', confidence=0.99,
            reason='No safety violations detected'
        )

    def check_output(self, response: str, context: List[str] = None) -> SafetyCheckResult:
        """Check AI output for safety violations."""
        # Check for PII leakage
        for pii_type, pattern in self.PII_PATTERNS.items():
            if re.search(pattern, response):
                return SafetyCheckResult(
                    is_safe=False,
                    category='pii_leakage',
                    confidence=0.9,
                    reason=f'Response contains potential {pii_type}'
                )

        # Check for hallucination signals
        hedging_phrases = ['I think', 'I believe', 'probably', 'might be',
                           'I assume', 'not sure but']
        hedging_count = sum(1 for p in hedging_phrases if p.lower() in response.lower())
        if hedging_count >= 3:
            return SafetyCheckResult(
                is_safe=True,
                category='low_confidence',
                confidence=0.7,
                reason='Response contains multiple hedging phrases — possible hallucination'
            )

        return SafetyCheckResult(
            is_safe=True, category='clean', confidence=0.95,
            reason='Output passed safety checks'
        )


class HallucinationDetector:
    """Detect hallucinations by cross-referencing with retrieved context."""

    def __init__(self, similarity_threshold: float = 0.3):
        self.threshold = similarity_threshold

    def detect(self, response: str, contexts: List[str]) -> Dict:
        """Check if response claims are grounded in context."""
        if not contexts:
            return {
                'is_grounded': False,
                'grounding_score': 0.0,
                'ungrounded_claims': ['No context provided for verification'],
                'recommendation': 'Cannot verify — no context available'
            }

        # Split response into claims (sentences)
        claims = [s.strip() for s in re.split(r'[.!?]', response) if len(s.strip()) > 20]
        context_text = ' '.join(contexts).lower()

        grounded_claims = []
        ungrounded_claims = []

        for claim in claims:
            # Simple word overlap check (in production: use embeddings)
            claim_words = set(claim.lower().split())
            context_words = set(context_text.split())
            overlap = len(claim_words & context_words) / max(len(claim_words), 1)

            if overlap >= self.threshold:
                grounded_claims.append(claim)
            else:
                ungrounded_claims.append(claim)

        grounding_score = len(grounded_claims) / max(len(claims), 1)

        return {
            'is_grounded': grounding_score >= 0.7,
            'grounding_score': round(grounding_score, 3),
            'total_claims': len(claims),
            'grounded_count': len(grounded_claims),
            'ungrounded_count': len(ungrounded_claims),
            'ungrounded_claims': ungrounded_claims[:5],
            'recommendation': 'Response is well-grounded' if grounding_score >= 0.7
                             else 'Response may contain hallucinations — review needed'
        }


class RAGASEvaluator:
    """Evaluate RAG pipeline quality using RAGAS-inspired metrics."""

    def evaluate(self, query: str, response: str, contexts: List[str],
                 ground_truth: str = None) -> RAGASMetrics:
        """Compute RAGAS metrics for a single query-response pair."""

        faithfulness = self._compute_faithfulness(response, contexts)
        relevancy = self._compute_answer_relevancy(query, response)
        precision = self._compute_context_precision(query, contexts)
        recall = self._compute_context_recall(response, contexts, ground_truth)

        overall = (faithfulness * 0.3 + relevancy * 0.3 +
                   precision * 0.2 + recall * 0.2)

        return RAGASMetrics(
            faithfulness=round(faithfulness, 3),
            answer_relevancy=round(relevancy, 3),
            context_precision=round(precision, 3),
            context_recall=round(recall, 3),
            overall_score=round(overall, 3)
        )

    def _compute_faithfulness(self, response: str, contexts: List[str]) -> float:
        """How much of the response is supported by context."""
        if not contexts:
            return 0.0
        response_words = set(response.lower().split())
        context_words = set(' '.join(contexts).lower().split())
        overlap = len(response_words & context_words)
        return min(overlap / max(len(response_words), 1) * 2, 1.0)

    def _compute_answer_relevancy(self, query: str, response: str) -> float:
        """How relevant the response is to the query."""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words & response_words)
        return min(overlap / max(len(query_words), 1) * 1.5, 1.0)

    def _compute_context_precision(self, query: str, contexts: List[str]) -> float:
        """How precise the retrieved contexts are."""
        if not contexts:
            return 0.0
        query_words = set(query.lower().split())
        relevant = 0
        for ctx in contexts:
            ctx_words = set(ctx.lower().split())
            if len(query_words & ctx_words) / max(len(query_words), 1) > 0.2:
                relevant += 1
        return relevant / len(contexts)

    def _compute_context_recall(self, response: str, contexts: List[str],
                                 ground_truth: str = None) -> float:
        """How much of the needed information was retrieved."""
        if ground_truth:
            gt_words = set(ground_truth.lower().split())
            ctx_words = set(' '.join(contexts).lower().split())
            return min(len(gt_words & ctx_words) / max(len(gt_words), 1) * 1.5, 1.0)
        return 0.8  # Default when no ground truth


class GuardrailsPipeline:
    """Complete guardrails pipeline: input check → processing → output check."""

    def __init__(self):
        self.safety = ContentSafetyGuard()
        self.hallucination = HallucinationDetector()
        self.evaluator = RAGASEvaluator()

    def process(self, query: str, response: str, contexts: List[str] = None,
                ground_truth: str = None) -> Dict:
        """Run full guardrails pipeline."""
        # Input safety
        input_check = self.safety.check_input(query)
        if not input_check.is_safe:
            return {
                'status': 'blocked',
                'stage': 'input_safety',
                'reason': input_check.reason,
                'response': None
            }

        # Output safety
        output_check = self.safety.check_output(response, contexts)

        # Hallucination check
        hallucination = self.hallucination.detect(response, contexts or [])

        # Quality evaluation
        metrics = self.evaluator.evaluate(query, response, contexts or [], ground_truth)

        return {
            'status': 'passed' if output_check.is_safe and hallucination['is_grounded'] else 'flagged',
            'input_safety': input_check.__dict__,
            'output_safety': output_check.__dict__,
            'hallucination': hallucination,
            'quality_metrics': metrics.__dict__,
            'response': response if output_check.is_safe else '[REDACTED — safety violation]'
        }


if __name__ == '__main__':
    pipeline = GuardrailsPipeline()

    # Test case
    query = "What is our code review process?"
    response = "All code must pass peer review before merging. Reviews should check for correctness, performance, and test coverage of at least 80%."
    contexts = ["Code Review Guidelines: All code must pass peer review before merging. Reviews should check for correctness, performance implications, security vulnerabilities, and test coverage minimum 80%."]

    result = pipeline.process(query, response, contexts)
    print(f"\nStatus: {result['status']}")
    print(f"Grounded: {result['hallucination']['is_grounded']}")
    print(f"Quality: {result['quality_metrics']}")
