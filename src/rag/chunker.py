"""
Semantic Chunker — splits documents at natural boundaries rather than fixed token counts.
Preserves context, headings, and paragraph structure for better RAG retrieval.
"""

import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    content: str
    tokens: int
    chunk_index: int
    metadata: Dict


class SemanticChunker:
    """
    Splits documents using semantic boundaries: headings, paragraphs, sentences.
    Supports overlap for context continuity across chunks.
    """

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64,
                 strategy: str = 'semantic'):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.strategy = strategy

    def chunk(self, text: str, doc_id: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """Split a document into chunks using the configured strategy."""
        if self.strategy == 'semantic':
            raw_chunks = self._semantic_split(text)
        elif self.strategy == 'sentence':
            raw_chunks = self._sentence_split(text)
        else:
            raw_chunks = self._fixed_split(text)

        chunks = []
        for i, content in enumerate(raw_chunks):
            content = content.strip()
            if not content:
                continue
            chunks.append(Chunk(
                chunk_id=f"{doc_id}_chunk_{i:04d}",
                doc_id=doc_id,
                content=content,
                tokens=self._count_tokens(content),
                chunk_index=i,
                metadata=metadata or {}
            ))

        logger.debug(f"Chunked doc {doc_id}: {len(chunks)} chunks")
        return chunks

    def _semantic_split(self, text: str) -> List[str]:
        """Split on headings and paragraph boundaries."""
        # Split on markdown headings
        heading_pattern = r'(?=#{1,3}\s)'
        sections = re.split(heading_pattern, text)

        chunks = []
        for section in sections:
            if self._count_tokens(section) <= self.max_tokens:
                chunks.append(section)
            else:
                # Further split long sections by paragraph
                paragraphs = re.split(r'\n\n+', section)
                buffer = ''
                for para in paragraphs:
                    if self._count_tokens(buffer + para) <= self.max_tokens:
                        buffer += '\n\n' + para
                    else:
                        if buffer:
                            chunks.append(buffer.strip())
                        buffer = para
                if buffer:
                    chunks.append(buffer.strip())

        return self._add_overlap(chunks)

    def _sentence_split(self, text: str) -> List[str]:
        """Split by sentences, grouping into max_token windows."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, buffer = [], ''
        for sent in sentences:
            if self._count_tokens(buffer + sent) <= self.max_tokens:
                buffer += ' ' + sent
            else:
                if buffer:
                    chunks.append(buffer.strip())
                buffer = sent
        if buffer:
            chunks.append(buffer.strip())
        return self._add_overlap(chunks)

    def _fixed_split(self, text: str) -> List[str]:
        """Simple fixed-size word splitting with overlap."""
        words = text.split()
        step = max(1, self.max_tokens - self.overlap_tokens)
        return [' '.join(words[i:i + self.max_tokens])
                for i in range(0, len(words), step)]

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap context between adjacent chunks."""
        if self.overlap_tokens == 0 or len(chunks) <= 1:
            return chunks
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_words = chunks[i-1].split()[-self.overlap_tokens:]
            overlap_text = ' '.join(prev_words)
            overlapped.append(overlap_text + '\n' + chunks[i])
        return overlapped

    def _count_tokens(self, text: str) -> int:
        # Approximate: 1 token ≈ 4 chars (tiktoken would be more accurate)
        return max(1, len(text) // 4)

    def batch_chunk(self, documents: List[Dict]) -> List[Chunk]:
        """Chunk a list of documents."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk(doc['content'], doc['doc_id'], doc.get('metadata', {}))
            all_chunks.extend(chunks)
        logger.info(f"Batch chunked {len(documents)} docs → {len(all_chunks)} chunks")
        return all_chunks


if __name__ == '__main__':
    chunker = SemanticChunker(max_tokens=256, overlap_tokens=32, strategy='semantic')

    sample_doc = """# Enterprise AI Strategy

## Executive Summary
Our enterprise AI initiative aims to deploy multi-agent systems across all business units.
This will reduce manual work by 40% and improve decision quality significantly.

## Phase 1: Foundation
Deploy RAG-powered knowledge bases for HR, Legal, and Finance.
Integrate with existing Confluence and SharePoint systems.

## Phase 2: Agents
Roll out specialized agents: Code, Data, Document, and Research.
Each agent will be fine-tuned on domain-specific data.

## Phase 3: Orchestration
Deploy the master orchestrator to route tasks automatically.
Implement feedback loops and continuous learning pipelines.
"""

    chunks = chunker.chunk(sample_doc, 'doc_001', {'source': 'strategy_doc', 'team': 'AI'})
    print(f"Generated {len(chunks)} chunks")
    for c in chunks:
        print(f"\nChunk {c.chunk_index} ({c.tokens} tokens):")
        print(c.content[:150] + '...' if len(c.content) > 150 else c.content)
