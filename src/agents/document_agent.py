"""
Document Agent — drafts reports, emails, compliance documents, and meeting summaries.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentType(Enum):
    EMAIL = "email"
    REPORT = "report"
    MEETING_SUMMARY = "meeting_summary"
    COMPLIANCE = "compliance"
    PROPOSAL = "proposal"
    EXECUTIVE_BRIEF = "executive_brief"


@dataclass
class DocumentRequest:
    doc_type: DocumentType
    topic: str
    context: str = ''
    audience: str = 'general'
    tone: str = 'professional'
    length: str = 'medium'    # short, medium, long
    key_points: List[str] = field(default_factory=list)


@dataclass
class Document:
    doc_type: DocumentType
    title: str
    content: str
    word_count: int
    created_at: str
    metadata: Dict


TEMPLATES = {
    DocumentType.EMAIL: """Subject: {topic}

Dear {audience},

{opening}

{body}

{closing}

Best regards,
[Your Name]""",

    DocumentType.REPORT: """# {topic}
**Date:** {date} | **Prepared for:** {audience}

---

## Executive Summary
{summary}

## Key Findings
{findings}

## Analysis
{analysis}

## Recommendations
{recommendations}

## Conclusion
{conclusion}""",

    DocumentType.MEETING_SUMMARY: """# Meeting Summary — {topic}
**Date:** {date} | **Attendees:** {audience}

---

## Discussion Points
{discussion}

## Decisions Made
{decisions}

## Action Items
{action_items}

## Next Steps
{next_steps}""",
}


class DocumentAgent:
    """Drafts professional documents using templates and LLM-powered content generation."""

    def __init__(self, model: str = 'gpt-4o'):
        self.model = model
        self.documents: List[Document] = []
        logger.info(f"Document Agent initialized with model={model}")

    def draft(self, request: DocumentRequest) -> Document:
        """Generate a document based on the request."""
        logger.info(f"Drafting {request.doc_type.value}: {request.topic[:60]}")

        content = self._generate_content(request)
        title = f"{request.doc_type.value.replace('_', ' ').title()} — {request.topic}"

        doc = Document(
            doc_type=request.doc_type,
            title=title,
            content=content,
            word_count=len(content.split()),
            created_at=datetime.now().isoformat(),
            metadata={
                'audience': request.audience,
                'tone': request.tone,
                'length': request.length,
                'model': self.model
            }
        )
        self.documents.append(doc)
        logger.info(f"Document drafted — {doc.word_count} words")
        return doc

    def _generate_content(self, req: DocumentRequest) -> str:
        date = datetime.now().strftime('%B %d, %Y')
        points = '\n'.join(f"- {p}" for p in req.key_points) if req.key_points else f"- Key point about {req.topic}"

        if req.doc_type == DocumentType.EMAIL:
            opening = f"I am writing to discuss {req.topic}."
            body = req.context or f"Following our recent discussions on {req.topic}, I wanted to share key updates."
            closing = "Please let me know if you have any questions or would like to schedule a call."
            return TEMPLATES[DocumentType.EMAIL].format(
                topic=req.topic, audience=req.audience,
                opening=opening, body=body, closing=closing
            )

        elif req.doc_type == DocumentType.REPORT:
            return TEMPLATES[DocumentType.REPORT].format(
                topic=req.topic, date=date, audience=req.audience,
                summary=f"This report covers {req.topic}. {req.context[:200] if req.context else ''}",
                findings=points,
                analysis=f"Analysis of {req.topic} reveals significant patterns that require attention.",
                recommendations=f"Based on the findings, we recommend:\n{points}",
                conclusion=f"In conclusion, {req.topic} requires strategic focus going forward."
            )

        elif req.doc_type == DocumentType.MEETING_SUMMARY:
            return TEMPLATES[DocumentType.MEETING_SUMMARY].format(
                topic=req.topic, date=date, audience=req.audience,
                discussion=points,
                decisions=f"- Agreed to proceed with {req.topic}\n- Assigned owners for each workstream",
                action_items=f"- [ ] Follow up on {req.topic} — Owner: TBD — Due: Next sprint",
                next_steps=f"- Schedule follow-up meeting\n- Share this summary with stakeholders"
            )

        else:
            return (f"# {req.topic}\n\n"
                    f"**Date:** {date}\n\n"
                    f"{req.context}\n\n"
                    f"## Key Points\n{points}\n")

    def revise(self, doc: Document, feedback: str) -> Document:
        """Revise a document based on feedback."""
        logger.info(f"Revising document: {doc.title[:50]} — feedback: {feedback[:60]}")
        revised_content = doc.content + f"\n\n---\n*Revised based on feedback: {feedback}*"
        doc.content = revised_content
        doc.word_count = len(revised_content.split())
        return doc

    def stats(self) -> Dict:
        if not self.documents:
            return {'documents_drafted': 0}
        type_counts = {}
        for d in self.documents:
            type_counts[d.doc_type.value] = type_counts.get(d.doc_type.value, 0) + 1
        return {
            'documents_drafted': len(self.documents),
            'avg_word_count': round(sum(d.word_count for d in self.documents) / len(self.documents)),
            'by_type': type_counts
        }


if __name__ == '__main__':
    agent = DocumentAgent()

    requests = [
        DocumentRequest(DocumentType.EMAIL, 'Q4 Budget Review',
                        audience='Finance Team', key_points=['15% over budget', 'Need approval']),
        DocumentRequest(DocumentType.REPORT, 'AI Copilot ROI Analysis',
                        audience='Executive Leadership', context='3-month pilot results',
                        key_points=['40% productivity gain', '$2M cost savings', '93% user satisfaction']),
        DocumentRequest(DocumentType.MEETING_SUMMARY, 'Product Roadmap Planning',
                        audience='Engineering + Product'),
    ]

    for req in requests:
        doc = agent.draft(req)
        print(f"\n{'='*60}")
        print(f"Type: {doc.doc_type.value} | Words: {doc.word_count}")
        print(doc.content[:300] + '...')

    print(f"\nStats: {agent.stats()}")
