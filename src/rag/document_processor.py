"""
Document Processing & Semantic Chunking Pipeline
Multi-format parsing with intelligent semantic chunking for RAG.
"""

import os
import re
import hashlib
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A semantically meaningful chunk of a document."""
    chunk_id: str
    doc_id: str
    content: str
    metadata: Dict
    token_count: int
    chunk_index: int
    source_file: str
    doc_type: str


class DocumentProcessor:
    """Multi-format document parser for enterprise docs."""

    SUPPORTED_FORMATS = {'.pdf', '.docx', '.txt', '.md', '.html', '.csv', '.json', '.py', '.js', '.ts'}

    def __init__(self):
        self.processed_count = 0

    def parse_document(self, file_path: str) -> Dict:
        """Parse any supported document format into raw text + metadata."""
        ext = os.path.splitext(file_path)[1].lower()
        doc_id = hashlib.md5(file_path.encode()).hexdigest()

        metadata = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_type': ext,
            'file_size_kb': round(os.path.getsize(file_path) / 1024, 2) if os.path.exists(file_path) else 0,
            'processed_at': datetime.now().isoformat(),
            'doc_id': doc_id,
        }

        if ext == '.txt' or ext == '.md':
            text = self._read_text(file_path)
        elif ext == '.pdf':
            text = self._parse_pdf(file_path)
        elif ext == '.docx':
            text = self._parse_docx(file_path)
        elif ext == '.html':
            text = self._parse_html(file_path)
        elif ext in {'.py', '.js', '.ts', '.json', '.csv'}:
            text = self._read_text(file_path)
            metadata['is_code'] = ext in {'.py', '.js', '.ts'}
        else:
            logger.warning(f"Unsupported format: {ext}")
            text = ""

        metadata['char_count'] = len(text)
        metadata['word_count'] = len(text.split())
        self.processed_count += 1

        return {'text': text, 'metadata': metadata, 'doc_id': doc_id}

    def _read_text(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            return ""

    def _parse_pdf(self, path: str) -> str:
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text_parts.append(page.extract_text() or "")
            return "\n\n".join(text_parts)
        except ImportError:
            logger.warning("pdfplumber not installed. Using fallback.")
            return f"[PDF content from {path}]"

    def _parse_docx(self, path: str) -> str:
        try:
            from docx import Document
            doc = Document(path)
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            return f"[DOCX content from {path}]"

    def _parse_html(self, path: str) -> str:
        try:
            from bs4 import BeautifulSoup
            with open(path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            return soup.get_text(separator='\n', strip=True)
        except Exception:
            return self._read_text(path)


class SemanticChunker:
    """Intelligent document chunking using semantic boundaries."""

    def __init__(self, max_chunk_size: int = 512, overlap: int = 50,
                 min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, doc: Dict) -> List[DocumentChunk]:
        """Split document into semantic chunks."""
        text = doc['text']
        metadata = doc['metadata']
        doc_id = doc['doc_id']

        if not text.strip():
            return []

        # Try semantic splitting first (headers, paragraphs)
        sections = self._split_by_semantics(text)

        chunks = []
        chunk_idx = 0
        for section in sections:
            if len(section.split()) <= self.max_chunk_size:
                if len(section.split()) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(
                        section, doc_id, metadata, chunk_idx
                    ))
                    chunk_idx += 1
            else:
                # Further split large sections with overlap
                sub_chunks = self._split_with_overlap(section)
                for sub in sub_chunks:
                    chunks.append(self._create_chunk(
                        sub, doc_id, metadata, chunk_idx
                    ))
                    chunk_idx += 1

        logger.debug(f"Document {metadata['file_name']}: {len(chunks)} chunks")
        return chunks

    def _split_by_semantics(self, text: str) -> List[str]:
        """Split text at semantic boundaries (headers, double newlines)."""
        # Split on markdown headers
        header_pattern = r'\n(?=#{1,4}\s)'
        sections = re.split(header_pattern, text)

        # Further split on double newlines
        result = []
        for section in sections:
            paragraphs = re.split(r'\n\s*\n', section)
            result.extend([p.strip() for p in paragraphs if p.strip()])

        return result

    def _split_with_overlap(self, text: str) -> List[str]:
        """Split text into overlapping windows."""
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = min(start + self.max_chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start = end - self.overlap

        return chunks

    def _create_chunk(self, text: str, doc_id: str, metadata: Dict,
                      chunk_idx: int) -> DocumentChunk:
        """Create a DocumentChunk object."""
        chunk_id = hashlib.md5(f"{doc_id}_{chunk_idx}_{text[:50]}".encode()).hexdigest()
        return DocumentChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            content=text,
            metadata={**metadata, 'chunk_index': chunk_idx},
            token_count=len(text.split()),
            chunk_index=chunk_idx,
            source_file=metadata.get('file_name', ''),
            doc_type=metadata.get('file_type', '')
        )


class DocumentPipeline:
    """End-to-end document processing pipeline."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.processor = DocumentProcessor()
        self.chunker = SemanticChunker(max_chunk_size=chunk_size, overlap=overlap)

    def process_directory(self, dir_path: str) -> List[DocumentChunk]:
        """Process all documents in a directory."""
        logger.info(f"Processing documents from: {dir_path}")
        all_chunks = []

        if not os.path.exists(dir_path):
            logger.info("Directory not found. Generating sample documents...")
            self._generate_sample_docs(dir_path)

        for root, dirs, files in os.walk(dir_path):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in DocumentProcessor.SUPPORTED_FORMATS:
                    fpath = os.path.join(root, fname)
                    try:
                        doc = self.processor.parse_document(fpath)
                        chunks = self.chunker.chunk_document(doc)
                        all_chunks.extend(chunks)
                    except Exception as e:
                        logger.error(f"Error processing {fname}: {e}")

        logger.info(f"\n✅ Processed {self.processor.processed_count} documents → {len(all_chunks)} chunks")
        return all_chunks

    def _generate_sample_docs(self, dir_path: str):
        """Generate sample enterprise documents for demo."""
        os.makedirs(dir_path, exist_ok=True)

        docs = {
            'engineering_standards.md': """# Engineering Standards & Best Practices

## Code Review Guidelines
All code must pass peer review before merging. Reviews should check for:
- Correctness and edge cases
- Performance implications
- Security vulnerabilities
- Test coverage (minimum 80%)
- Documentation completeness

## API Design Standards
REST APIs must follow OpenAPI 3.0 specification. All endpoints require authentication.
Rate limiting: 1000 req/min for standard, 10000 req/min for premium.

## Database Migration Policy
All schema changes require a migration file. Breaking changes need 2-sprint deprecation.
Always test migrations on staging before production deployment.""",

            'onboarding_guide.md': """# New Employee Onboarding Guide

## Week 1: Setup & Orientation
- IT setup: laptop, email, Slack, Jira, Confluence, GitHub access
- HR orientation: benefits enrollment, company policies, org chart
- Meet your team lead and buddy

## Week 2: Technical Deep Dive
- Architecture overview session with lead engineer
- Codebase walkthrough and first PR
- Access production monitoring dashboards

## Week 3: First Project
- Pick up a starter ticket from the backlog
- Attend sprint planning and standup
- Complete first code review cycle""",

            'incident_response.md': """# Incident Response Playbook

## Severity Levels
- P0 (Critical): Full service outage. Response time: 15 min. All hands.
- P1 (High): Major feature degraded. Response time: 30 min.
- P2 (Medium): Minor feature issue. Response time: 4 hours.
- P3 (Low): Cosmetic/non-urgent. Response time: Next sprint.

## Escalation Chain
1. On-call engineer acknowledges alert
2. Assess severity and impact
3. Page incident commander if P0/P1
4. Open bridge call, post in #incidents
5. Resolve → Post-mortem within 48 hours""",

            'ml_deployment_guide.md': """# ML Model Deployment Guide

## Model Registry
All models must be registered in MLflow with:
- Model artifact (ONNX/PyTorch/TF)
- Training metrics and evaluation results
- Data lineage and feature importance
- A/B test configuration

## Serving Infrastructure
- Real-time: SageMaker endpoints with auto-scaling
- Batch: Airflow DAGs → SageMaker Batch Transform
- Edge: ONNX Runtime with TensorRT optimization

## Monitoring
Track: prediction latency, throughput, data drift, model accuracy decay.
Alert thresholds: >200ms P99 latency, >5% accuracy drop, >0.1 PSI drift score.""",

            'data_privacy_policy.md': """# Data Privacy & Compliance Policy

## PII Handling
- All PII must be encrypted at rest (AES-256) and in transit (TLS 1.3)
- PII access requires explicit role-based authorization
- Data retention: 90 days for logs, 7 years for financial records
- Right to deletion: Process within 30 days of request

## GDPR Compliance
- Consent management for EU users
- Data Processing Agreements with all vendors
- Annual privacy impact assessments
- Breach notification within 72 hours"""
        }

        for fname, content in docs.items():
            with open(os.path.join(dir_path, fname), 'w') as f:
                f.write(content)

        logger.info(f"Generated {len(docs)} sample documents in {dir_path}")


if __name__ == '__main__':
    pipeline = DocumentPipeline(chunk_size=512, overlap=50)
    chunks = pipeline.process_directory('data/documents')
    print(f"\nSample chunk:")
    if chunks:
        c = chunks[0]
        print(f"  ID: {c.chunk_id}")
        print(f"  Source: {c.source_file}")
        print(f"  Tokens: {c.token_count}")
        print(f"  Content: {c.content[:200]}...")
