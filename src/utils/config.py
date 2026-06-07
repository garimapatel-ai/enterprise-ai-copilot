"""Central configuration for the Enterprise AI Copilot."""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class RAGConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunking_strategy: str = 'semantic'
    embedding_model: str = 'text-embedding-3-large'
    embedding_dim: int = 1536
    vector_db: str = 'pinecone'
    top_k_retrieval: int = 20
    top_n_rerank: int = 5
    rrf_k: int = 60
    entailment_threshold: float = 0.3


@dataclass
class LLMConfig:
    default_model: str = 'gpt-4o'
    fallback_model: str = 'claude-3-5-sonnet'
    max_tokens: int = 2048
    temperature: float = 0.1
    budget_per_request: float = 0.10
    max_latency_ms: float = 2000


@dataclass
class AgentConfig:
    max_iterations: int = 10
    memory_window: int = 20
    tool_timeout_s: int = 30
    agents: List[str] = field(default_factory=lambda: [
        'code', 'data', 'document', 'research', 'orchestrator'
    ])


@dataclass
class GuardrailsConfig:
    enable_safety_check: bool = True
    enable_hallucination_detection: bool = True
    enable_ragas_eval: bool = True
    hallucination_threshold: float = 0.3
    safety_categories: List[str] = field(default_factory=lambda: [
        'harmful_content', 'pii', 'proprietary_data', 'policy_violation'
    ])


@dataclass
class APIConfig:
    host: str = '0.0.0.0'
    port: int = 8000
    workers: int = 4
    rate_limit: int = 100
    jwt_expiry_hours: int = 24
    cors_origins: List[str] = field(default_factory=lambda: ['http://localhost:3000'])


@dataclass
class AppConfig:
    rag: RAGConfig = field(default_factory=RAGConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    guardrails: GuardrailsConfig = field(default_factory=GuardrailsConfig)
    api: APIConfig = field(default_factory=APIConfig)
    mlflow_uri: str = 'mlruns'
    wandb_project: str = 'enterprise-ai-copilot'
    seed: int = 42


CONFIG = AppConfig()
