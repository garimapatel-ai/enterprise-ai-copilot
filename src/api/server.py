"""
FastAPI Backend Server
REST API for the Enterprise AI Copilot.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enterprise AI Copilot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    use_rag: bool = True
    top_k: int = 5
    model: str = "gpt-4o"


class QueryResponse(BaseModel):
    task_id: str
    query: str
    response: str
    sources: List[dict]
    agent_used: str
    metrics: dict


# Initialize components (lazy loading in production)
retriever = None
orchestrator = None
guardrails = None


def get_retriever():
    global retriever
    if retriever is None:
        from rag.document_processor import DocumentPipeline
        from rag.retriever import HybridRetriever
        pipeline = DocumentPipeline()
        chunks = pipeline.process_directory('data/documents')
        retriever = HybridRetriever()
        retriever.index_chunks(chunks)
    return retriever


def get_orchestrator():
    global orchestrator
    if orchestrator is None:
        from agents.orchestrator import MasterOrchestrator
        orchestrator = MasterOrchestrator()
    return orchestrator


def get_guardrails():
    global guardrails
    if guardrails is None:
        from guardrails.safety_checker import GuardrailsPipeline
        guardrails = GuardrailsPipeline()
    return guardrails


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query through the AI copilot pipeline."""
    try:
        orch = get_orchestrator()
        guard = get_guardrails()

        # RAG retrieval
        contexts = []
        sources = []
        if request.use_rag:
            ret = get_retriever()
            results = ret.retrieve(request.query, top_k=request.top_k)
            contexts = [r.content for r in results]
            sources = [{'source': r.metadata.get('file_name', 'N/A'),
                       'score': r.final_score} for r in results]

        # Agent execution
        result = orch.execute(request.query, rag_context=contexts)
        response_text = f"Based on enterprise knowledge: {contexts[0][:200]}..." if contexts \
            else "I can help with that. Let me process your request."

        # Guardrails check
        safety = guard.process(request.query, response_text, contexts)

        return QueryResponse(
            task_id=result['task_id'],
            query=request.query,
            response=safety.get('response', response_text),
            sources=sources,
            agent_used=result['agent_results'][0]['agent'] if result['agent_results'] else 'orchestrator',
            metrics=safety.get('quality_metrics', {})
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index")
async def index_documents(directory: str = "data/documents"):
    """Index documents for RAG retrieval."""
    try:
        from rag.document_processor import DocumentPipeline
        pipeline = DocumentPipeline()
        chunks = pipeline.process_directory(directory)

        global retriever
        from rag.retriever import HybridRetriever
        retriever = HybridRetriever()
        retriever.index_chunks(chunks)

        return {"status": "indexed", "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
