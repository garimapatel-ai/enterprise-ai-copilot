"""
Hallucination Detector — cross-references LLM responses against source documents.
Uses entailment scoring and factual consistency checks.
"""

import re
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HallucinationReport:
    response: str
    is_grounded: bool
    confidence: float
    grounded_claims: List[str]
    ungrounded_claims: List[str]
    risk_level: str   # low, medium, high
    recommendation: str


class ClaimExtractor:
    """Extracts factual claims from LLM responses for verification."""

    CLAIM_PATTERNS = [
        r'\d+[\.,]?\d*\s*%',             # percentages
        r'\$[\d,]+',                       # dollar amounts
        r'\d{4}',                          # years
        r'(?:increased?|decreased?|grew?|fell?|rose?)\s+by\s+[\d\.]+',
        r'(?:always|never|all|every|none)\s+\w+',
    ]

    def extract(self, text: str) -> List[str]:
        """Extract sentences containing quantitative or absolute claims."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        claims = []
        for sent in sentences:
            if any(re.search(pat, sent, re.IGNORECASE) for pat in self.CLAIM_PATTERNS):
                claims.append(sent.strip())
        return claims


class EntailmentScorer:
    """Scores whether a claim is entailed by source documents."""

    def score(self, claim: str, sources: List[str]) -> Tuple[float, str]:
        """
        Returns (entailment_score, best_matching_source).
        In production: replace with NLI model (e.g. deberta-v3-large-mnli).
        """
        if not sources:
            return 0.0, ''

        best_score = 0.0
        best_source = ''

        claim_tokens = set(claim.lower().split())
        for src in sources:
            src_tokens = set(src.lower().split())
            overlap = len(claim_tokens & src_tokens)
            score = overlap / (len(claim_tokens) + 1e-8)

            # Bonus for numeric match
            claim_nums = set(re.findall(r'\d+\.?\d*', claim))
            src_nums = set(re.findall(r'\d+\.?\d*', src))
            if claim_nums & src_nums:
                score += 0.3

            if score > best_score:
                best_score = score
                best_source = src[:100]

        return round(min(1.0, best_score), 3), best_source


class HallucinationDetector:
    """
    Detects hallucinations by verifying LLM response claims against retrieved context.
    Part of the RAG guardrails pipeline.
    """

    def __init__(self, entailment_threshold: float = 0.3):
        self.threshold = entailment_threshold
        self.extractor = ClaimExtractor()
        self.scorer = EntailmentScorer()
        self.detection_log: List[HallucinationReport] = []

    def check(self, response: str, source_documents: List[str]) -> HallucinationReport:
        """Check a response for hallucinations given the source context."""
        claims = self.extractor.extract(response)

        if not claims:
            report = HallucinationReport(
                response=response, is_grounded=True, confidence=0.8,
                grounded_claims=[], ungrounded_claims=[],
                risk_level='low', recommendation="No factual claims detected — response appears safe."
            )
            self.detection_log.append(report)
            return report

        grounded, ungrounded = [], []
        scores = []

        for claim in claims:
            score, _ = self.scorer.score(claim, source_documents)
            scores.append(score)
            if score >= self.threshold:
                grounded.append(claim)
            else:
                ungrounded.append(claim)

        avg_score = sum(scores) / len(scores)
        hallucination_rate = len(ungrounded) / len(claims)
        is_grounded = hallucination_rate < 0.3

        if hallucination_rate == 0:
            risk_level = 'low'
            recommendation = "All claims are supported by source documents."
        elif hallucination_rate < 0.3:
            risk_level = 'medium'
            recommendation = f"Review {len(ungrounded)} potentially ungrounded claim(s) before sending."
        else:
            risk_level = 'high'
            recommendation = "Response contains significant unsupported claims — regenerate with stricter grounding."

        report = HallucinationReport(
            response=response, is_grounded=is_grounded,
            confidence=round(avg_score, 3),
            grounded_claims=grounded, ungrounded_claims=ungrounded,
            risk_level=risk_level, recommendation=recommendation
        )
        self.detection_log.append(report)
        logger.info(f"Hallucination check — risk: {risk_level}, grounded: {len(grounded)}/{len(claims)}")
        return report

    def stats(self) -> Dict:
        if not self.detection_log:
            return {'checks': 0}
        high_risk = sum(1 for r in self.detection_log if r.risk_level == 'high')
        return {
            'total_checks': len(self.detection_log),
            'high_risk_rate': f"{high_risk/len(self.detection_log):.1%}",
            'avg_confidence': round(sum(r.confidence for r in self.detection_log) / len(self.detection_log), 3),
        }


if __name__ == '__main__':
    detector = HallucinationDetector(entailment_threshold=0.3)

    sources = [
        "Our AI copilot achieved 96.2% factual accuracy in Q3 2024 trials.",
        "Response latency averaged 1.8 seconds across 10,000 test queries.",
        "The system indexed 500,000 internal documents from Confluence and SharePoint.",
    ]

    responses = [
        "The AI copilot achieved 96.2% accuracy and processed queries in 1.8 seconds on average.",  # grounded
        "The system achieved 99.9% accuracy and processes queries in under 100ms with zero errors.",  # hallucinated
        "The copilot indexed enterprise documents and provides fast responses.",  # vague, no specific claims
    ]

    for resp in responses:
        report = detector.check(resp, sources)
        print(f"\nResponse: {resp[:70]}...")
        print(f"  Risk: {report.risk_level} | Grounded: {report.is_grounded} | Confidence: {report.confidence}")
        print(f"  {report.recommendation}")

    print(f"\nDetector stats: {detector.stats()}")
