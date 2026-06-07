"""
Research Agent — performs web research, synthesizes information, and generates reports.
Uses RAG over internal knowledge base + simulated web search.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ResearchQuery:
    query_id: str
    question: str
    depth: str = 'medium'       # shallow, medium, deep
    sources: List[str] = field(default_factory=lambda: ['internal_kb', 'web'])
    max_sources: int = 10


@dataclass
class Source:
    title: str
    url: str
    snippet: str
    relevance_score: float
    source_type: str


@dataclass
class ResearchResult:
    query_id: str
    question: str
    answer: str
    sources: List[Source]
    confidence: float
    key_facts: List[str]
    limitations: List[str]
    generated_at: str


class ResearchAgent:
    """
    Research agent that retrieves, synthesizes, and cites information
    from internal knowledge bases and web sources.
    """

    # Simulated knowledge base entries
    KB_ENTRIES = {
        'ai': [
            ('Enterprise AI Adoption Report 2024', 'AI adoption in enterprises grew 42% YoY'),
            ('LLM Performance Benchmarks', 'GPT-4o achieves 96.3% on enterprise task benchmarks'),
            ('RAG System Design Guide', 'Hybrid retrieval improves factual accuracy by 23%'),
        ],
        'supply chain': [
            ('Supply Chain Risk Report', '$184B lost annually to supply chain disruptions'),
            ('Demand Forecasting Best Practices', 'TFT models reduce MAPE by 35% vs ARIMA baselines'),
        ],
        'cloud': [
            ('Azure Cost Optimization Guide', 'Right-sizing VMs reduces cloud spend by avg 31%'),
            ('Kubernetes Best Practices', 'Pod autoscaling reduces latency spikes by 67%'),
        ],
    }

    def __init__(self, model: str = 'gpt-4o', kb_path: Optional[str] = None):
        self.model = model
        self.kb_path = kb_path
        self.research_history: List[ResearchResult] = []
        logger.info(f"Research Agent initialized with model={model}")

    def research(self, query: ResearchQuery) -> ResearchResult:
        """Conduct research on a given question."""
        logger.info(f"Researching: {query.question[:80]}")

        sources = self._retrieve_sources(query)
        key_facts = self._extract_facts(sources)
        answer = self._synthesize_answer(query.question, key_facts, sources)
        confidence = self._compute_confidence(sources, key_facts)
        limitations = self._identify_limitations(sources, query)

        result = ResearchResult(
            query_id=query.query_id,
            question=query.question,
            answer=answer,
            sources=sources,
            confidence=confidence,
            key_facts=key_facts,
            limitations=limitations,
            generated_at=datetime.now().isoformat()
        )
        self.research_history.append(result)
        logger.info(f"Research complete — {len(sources)} sources, confidence: {confidence:.2f}")
        return result

    def _retrieve_sources(self, query: ResearchQuery) -> List[Source]:
        sources = []
        q_lower = query.question.lower()

        # Search internal KB
        for topic, entries in self.KB_ENTRIES.items():
            if topic in q_lower or any(w in q_lower for w in topic.split()):
                for title, snippet in entries[:2]:
                    sources.append(Source(
                        title=title,
                        url=f"internal://kb/{title.lower().replace(' ', '-')}",
                        snippet=snippet,
                        relevance_score=round(0.85 + 0.1 * (topic in q_lower), 3),
                        source_type='internal_kb'
                    ))

        # Simulated web sources
        if 'web' in query.sources:
            web_sources = [
                Source(f"Industry Report on {query.question[:30]}",
                       f"https://research.example.com/{query.query_id}",
                       f"Recent analysis on {query.question[:60]} shows significant trends.",
                       0.78, 'web'),
                Source(f"Expert Analysis: {query.question[:25]}",
                       f"https://insights.example.com/{query.query_id}",
                       f"Experts highlight key factors in {query.question[:40]}.",
                       0.72, 'web'),
            ]
            sources.extend(web_sources[:2])

        # Sort by relevance
        sources.sort(key=lambda s: s.relevance_score, reverse=True)
        return sources[:query.max_sources]

    def _extract_facts(self, sources: List[Source]) -> List[str]:
        facts = []
        for s in sources:
            if s.snippet:
                facts.append(f"{s.snippet} (Source: {s.title})")
        return facts[:8]

    def _synthesize_answer(self, question: str, facts: List[str], sources: List[Source]) -> str:
        n_internal = sum(1 for s in sources if s.source_type == 'internal_kb')
        n_web = len(sources) - n_internal

        answer = (
            f"Based on analysis of {len(sources)} sources ({n_internal} internal, {n_web} external):\n\n"
            f"**Question:** {question}\n\n"
            f"**Summary:** "
        )

        if facts:
            answer += facts[0] + "\n\n"
            if len(facts) > 1:
                answer += "**Supporting Evidence:**\n"
                answer += '\n'.join(f"• {f}" for f in facts[1:4])
        else:
            answer += f"No direct matches found in the knowledge base for '{question}'. "
            answer += "Consider expanding search scope or refining the query."

        return answer

    def _compute_confidence(self, sources: List[Source], facts: List[str]) -> float:
        if not sources:
            return 0.2
        avg_relevance = sum(s.relevance_score for s in sources) / len(sources)
        fact_bonus = min(0.1, len(facts) * 0.02)
        internal_bonus = 0.05 * sum(1 for s in sources if s.source_type == 'internal_kb')
        return round(min(0.98, avg_relevance + fact_bonus + internal_bonus), 3)

    def _identify_limitations(self, sources: List[Source], query: ResearchQuery) -> List[str]:
        limitations = []
        if len(sources) < 3:
            limitations.append("Limited source coverage — consider broadening search scope")
        if not any(s.source_type == 'internal_kb' for s in sources):
            limitations.append("No internal KB matches — answer based on external sources only")
        if query.depth == 'shallow':
            limitations.append("Shallow research mode — deeper analysis may reveal more nuance")
        return limitations

    def cite(self, result: ResearchResult) -> str:
        """Format citations for a research result."""
        lines = [f"## References for: {result.question[:60]}\n"]
        for i, s in enumerate(result.sources, 1):
            lines.append(f"[{i}] **{s.title}** ({s.source_type}) — relevance: {s.relevance_score}")
            lines.append(f"    {s.url}")
        return '\n'.join(lines)


if __name__ == '__main__':
    agent = ResearchAgent()

    queries = [
        ResearchQuery('r1', 'What are the best practices for RAG system design in enterprise AI?', depth='deep'),
        ResearchQuery('r2', 'How does AI impact supply chain demand forecasting accuracy?', depth='medium'),
        ResearchQuery('r3', 'What is the ROI of deploying LLMs in enterprise environments?', depth='medium'),
    ]

    for q in queries:
        result = agent.research(q)
        print(f"\n{'='*60}")
        print(f"Q: {result.question[:70]}")
        print(f"Confidence: {result.confidence} | Sources: {len(result.sources)}")
        print(result.answer[:400])
