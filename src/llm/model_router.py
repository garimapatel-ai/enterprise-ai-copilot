"""
Multi-LLM Router — routes tasks to the optimal model based on task type, cost, and latency.
Supports GPT-4o, Claude, Llama 3 70B, and Mistral 7B with fallback logic.
"""

import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    DATA_ANALYSIS = "data_analysis"
    DOCUMENT_DRAFTING = "document_drafting"
    RESEARCH = "research"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    CHAT = "chat"


@dataclass
class ModelConfig:
    name: str
    provider: str
    context_window: int
    cost_per_1k_tokens: float
    avg_latency_ms: float
    strengths: List[str]
    max_output_tokens: int = 4096


@dataclass
class RoutingDecision:
    task_type: TaskType
    selected_model: str
    reason: str
    estimated_cost: float
    estimated_latency_ms: float
    fallback_model: str


@dataclass
class LLMResponse:
    model: str
    content: str
    tokens_used: int
    latency_ms: float
    cost: float


MODELS = {
    'gpt-4o': ModelConfig(
        name='gpt-4o', provider='openai', context_window=128000,
        cost_per_1k_tokens=0.005, avg_latency_ms=800,
        strengths=['reasoning', 'code', 'analysis', 'instruction_following']
    ),
    'claude-3-5-sonnet': ModelConfig(
        name='claude-3-5-sonnet', provider='anthropic', context_window=200000,
        cost_per_1k_tokens=0.003, avg_latency_ms=700,
        strengths=['writing', 'summarization', 'document_drafting', 'long_context']
    ),
    'llama-3-70b': ModelConfig(
        name='llama-3-70b', provider='self_hosted', context_window=8192,
        cost_per_1k_tokens=0.0008, avg_latency_ms=1200,
        strengths=['cost_efficient', 'fine_tuning', 'domain_specific']
    ),
    'mistral-7b': ModelConfig(
        name='mistral-7b', provider='self_hosted', context_window=32768,
        cost_per_1k_tokens=0.0002, avg_latency_ms=400,
        strengths=['fast', 'classification', 'short_tasks', 'cost_efficient']
    ),
}

ROUTING_RULES = {
    TaskType.CODE_GENERATION: ('gpt-4o', 'llama-3-70b'),
    TaskType.DATA_ANALYSIS: ('gpt-4o', 'claude-3-5-sonnet'),
    TaskType.DOCUMENT_DRAFTING: ('claude-3-5-sonnet', 'gpt-4o'),
    TaskType.RESEARCH: ('gpt-4o', 'claude-3-5-sonnet'),
    TaskType.SUMMARIZATION: ('claude-3-5-sonnet', 'mistral-7b'),
    TaskType.CLASSIFICATION: ('mistral-7b', 'llama-3-70b'),
    TaskType.CHAT: ('gpt-4o', 'mistral-7b'),
}


class ModelRouter:
    """Routes tasks to the best LLM based on task type, budget, and latency constraints."""

    def __init__(self, budget_per_request: float = 0.10,
                 max_latency_ms: float = 2000,
                 prefer_cost: bool = False):
        self.budget = budget_per_request
        self.max_latency_ms = max_latency_ms
        self.prefer_cost = prefer_cost
        self.request_log: List[LLMResponse] = []
        logger.info(f"Model router initialized — budget=${budget_per_request}, latency<{max_latency_ms}ms")

    def route(self, task_type: TaskType, prompt_tokens: int = 500) -> RoutingDecision:
        """Select the best model for a given task."""
        primary, fallback = ROUTING_RULES.get(task_type, ('gpt-4o', 'mistral-7b'))
        model_cfg = MODELS[primary]

        # Check budget
        est_cost = (prompt_tokens / 1000) * model_cfg.cost_per_1k_tokens
        if est_cost > self.budget or (self.prefer_cost and MODELS[fallback].cost_per_1k_tokens < model_cfg.cost_per_1k_tokens):
            primary, fallback = fallback, primary
            model_cfg = MODELS[primary]
            est_cost = (prompt_tokens / 1000) * model_cfg.cost_per_1k_tokens

        # Check latency
        if model_cfg.avg_latency_ms > self.max_latency_ms:
            primary, fallback = fallback, primary
            model_cfg = MODELS[primary]

        decision = RoutingDecision(
            task_type=task_type,
            selected_model=primary,
            reason=f"Best match for {task_type.value}: strengths={model_cfg.strengths[:2]}",
            estimated_cost=round(est_cost, 6),
            estimated_latency_ms=model_cfg.avg_latency_ms,
            fallback_model=fallback
        )
        logger.info(f"Routing {task_type.value} → {primary} (est. ${est_cost:.5f})")
        return decision

    def call(self, task_type: TaskType, prompt: str,
             max_tokens: int = 512) -> LLMResponse:
        """Route and simulate an LLM call."""
        decision = self.route(task_type, len(prompt.split()))
        model_cfg = MODELS[decision.selected_model]

        # Simulate response
        t0 = time.time()
        response_text = self._simulate_response(task_type, prompt, decision.selected_model)
        latency = (time.time() - t0) * 1000 + model_cfg.avg_latency_ms

        tokens = len(prompt.split()) + len(response_text.split())
        cost = (tokens / 1000) * model_cfg.cost_per_1k_tokens

        result = LLMResponse(
            model=decision.selected_model,
            content=response_text,
            tokens_used=tokens,
            latency_ms=round(latency, 1),
            cost=round(cost, 6)
        )
        self.request_log.append(result)
        return result

    def _simulate_response(self, task_type: TaskType, prompt: str, model: str) -> str:
        responses = {
            TaskType.CODE_GENERATION: f"```python\ndef solution():\n    # Generated by {model}\n    pass\n```",
            TaskType.SUMMARIZATION: f"[{model}] Summary: {prompt[:100]}...",
            TaskType.CLASSIFICATION: f"[{model}] Classification: Category A (confidence: 0.94)",
            TaskType.DOCUMENT_DRAFTING: f"[{model}] Draft document based on: {prompt[:80]}",
            TaskType.DATA_ANALYSIS: f"[{model}] Analysis complete. Key finding: significant trend detected.",
            TaskType.RESEARCH: f"[{model}] Research synthesis: {prompt[:80]}",
            TaskType.CHAT: f"[{model}] {prompt[:50]}... Here's my response.",
        }
        return responses.get(task_type, f"[{model}] Response to: {prompt[:60]}")

    def usage_report(self) -> Dict:
        if not self.request_log:
            return {'total_requests': 0}
        total_cost = sum(r.cost for r in self.request_log)
        model_counts: Dict[str, int] = {}
        for r in self.request_log:
            model_counts[r.model] = model_counts.get(r.model, 0) + 1
        return {
            'total_requests': len(self.request_log),
            'total_cost_usd': round(total_cost, 4),
            'avg_latency_ms': round(sum(r.latency_ms for r in self.request_log) / len(self.request_log), 1),
            'model_usage': model_counts,
        }


if __name__ == '__main__':
    router = ModelRouter(budget_per_request=0.05, prefer_cost=False)

    tasks = [
        (TaskType.CODE_GENERATION, "Write a Python function to process CSV files"),
        (TaskType.DOCUMENT_DRAFTING, "Draft an executive summary for Q4 results"),
        (TaskType.CLASSIFICATION, "Classify this customer email as urgent or normal"),
        (TaskType.SUMMARIZATION, "Summarize this 10-page technical report"),
        (TaskType.DATA_ANALYSIS, "Analyze sales trends and identify anomalies"),
    ]

    for task_type, prompt in tasks:
        response = router.call(task_type, prompt)
        print(f"\n[{task_type.value}] → {response.model}")
        print(f"  Latency: {response.latency_ms:.0f}ms | Cost: ${response.cost:.5f}")
        print(f"  Response: {response.content[:80]}")

    print(f"\nUsage report: {router.usage_report()}")
