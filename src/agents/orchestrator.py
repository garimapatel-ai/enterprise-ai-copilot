"""
Multi-Agent Orchestrator
Master agent with specialized sub-agents for enterprise tasks.
"""

import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentRole(Enum):
    CODE = "code_agent"
    DATA = "data_agent"
    DOCUMENT = "document_agent"
    COMMUNICATION = "communication_agent"
    RESEARCH = "research_agent"


@dataclass
class AgentMessage:
    """Message between agents."""
    sender: str
    recipient: str
    content: str
    message_type: str  # task, result, question, error
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)


@dataclass
class TaskPlan:
    """Decomposed task plan from orchestrator."""
    task_id: str
    original_query: str
    steps: List[Dict]
    assigned_agents: List[str]
    requires_rag: bool
    estimated_complexity: str  # low, medium, high


class BaseAgent:
    """Base class for all specialized agents."""

    def __init__(self, name: str, role: AgentRole, capabilities: List[str]):
        self.name = name
        self.role = role
        self.capabilities = capabilities
        self.memory = []
        self.tools = []

    def process(self, task: str, context: Dict = None) -> Dict:
        """Process a task and return result."""
        raise NotImplementedError

    def can_handle(self, task_description: str) -> float:
        """Return confidence score (0-1) for handling this task."""
        task_lower = task_description.lower()
        score = sum(1 for cap in self.capabilities if cap in task_lower)
        return min(score / max(len(self.capabilities), 1), 1.0)


class CodeAgent(BaseAgent):
    """Specialized agent for code generation, review, and debugging."""

    def __init__(self):
        super().__init__(
            name="CodeBot",
            role=AgentRole.CODE,
            capabilities=['code', 'program', 'script', 'function', 'debug', 'review',
                          'refactor', 'test', 'api', 'bug', 'error', 'implement',
                          'python', 'javascript', 'sql', 'algorithm']
        )

    def process(self, task: str, context: Dict = None) -> Dict:
        """Generate, review, or debug code."""
        logger.info(f"[CodeBot] Processing: {task[:80]}...")

        if 'review' in task.lower():
            return self._review_code(task, context)
        elif 'debug' in task.lower() or 'fix' in task.lower():
            return self._debug_code(task, context)
        else:
            return self._generate_code(task, context)

    def _generate_code(self, task: str, context: Dict = None) -> Dict:
        prompt = f"""You are a senior software engineer. Generate clean, production-ready code.
Task: {task}
Context: {json.dumps(context or {})}
Include: type hints, docstrings, error handling, and unit tests."""
        return {
            'agent': self.name,
            'action': 'code_generation',
            'prompt': prompt,
            'status': 'ready_for_llm'
        }

    def _review_code(self, task: str, context: Dict = None) -> Dict:
        prompt = f"""Review this code for: correctness, performance, security, best practices.
{task}
Provide specific actionable feedback with line references."""
        return {
            'agent': self.name,
            'action': 'code_review',
            'prompt': prompt,
            'status': 'ready_for_llm'
        }

    def _debug_code(self, task: str, context: Dict = None) -> Dict:
        prompt = f"""Debug this code issue. Identify root cause, explain the bug, provide fix.
{task}
Show before/after code."""
        return {
            'agent': self.name,
            'action': 'debug',
            'prompt': prompt,
            'status': 'ready_for_llm'
        }


class DataAgent(BaseAgent):
    """Specialized agent for data analysis and visualization."""

    def __init__(self):
        super().__init__(
            name="DataBot",
            role=AgentRole.DATA,
            capabilities=['data', 'analyze', 'chart', 'graph', 'statistics', 'trend',
                          'dashboard', 'metric', 'report', 'csv', 'excel', 'sql',
                          'query', 'aggregate', 'forecast', 'visualization']
        )

    def process(self, task: str, context: Dict = None) -> Dict:
        logger.info(f"[DataBot] Processing: {task[:80]}...")
        prompt = f"""You are a senior data analyst. Analyze the data and provide insights.
Task: {task}
Include: statistical summary, key findings, visualizations (describe charts),
and actionable recommendations."""
        return {
            'agent': self.name,
            'action': 'data_analysis',
            'prompt': prompt,
            'status': 'ready_for_llm'
        }


class DocumentAgent(BaseAgent):
    """Specialized agent for document generation and processing."""

    def __init__(self):
        super().__init__(
            name="DocBot",
            role=AgentRole.DOCUMENT,
            capabilities=['document', 'report', 'draft', 'write', 'template',
                          'proposal', 'memo', 'summary', 'contract', 'compliance',
                          'policy', 'presentation', 'slides']
        )

    def process(self, task: str, context: Dict = None) -> Dict:
        logger.info(f"[DocBot] Processing: {task[:80]}...")
        prompt = f"""You are an expert technical writer. Create a professional document.
Task: {task}
Context: {json.dumps(context or {})}
Format with proper headings, clear structure, and professional tone."""
        return {
            'agent': self.name,
            'action': 'document_generation',
            'prompt': prompt,
            'status': 'ready_for_llm'
        }


class ResearchAgent(BaseAgent):
    """Specialized agent for research and information gathering."""

    def __init__(self):
        super().__init__(
            name="ResearchBot",
            role=AgentRole.RESEARCH,
            capabilities=['research', 'search', 'find', 'compare', 'evaluate',
                          'benchmark', 'investigate', 'explore', 'survey',
                          'literature', 'state of the art']
        )

    def process(self, task: str, context: Dict = None) -> Dict:
        logger.info(f"[ResearchBot] Processing: {task[:80]}...")
        prompt = f"""You are a thorough researcher. Investigate this topic comprehensively.
Task: {task}
Provide: key findings, source analysis, comparison of approaches,
and synthesis with recommendations."""
        return {
            'agent': self.name,
            'action': 'research',
            'prompt': prompt,
            'status': 'ready_for_llm'
        }


class MasterOrchestrator:
    """Master agent that routes tasks to specialized agents."""

    def __init__(self):
        self.agents = {
            AgentRole.CODE: CodeAgent(),
            AgentRole.DATA: DataAgent(),
            AgentRole.DOCUMENT: DocumentAgent(),
            AgentRole.RESEARCH: ResearchAgent(),
        }
        self.conversation_history = []
        self.task_log = []

    def plan_task(self, query: str) -> TaskPlan:
        """Decompose user query into a task plan."""
        # Score each agent's confidence
        scores = {}
        for role, agent in self.agents.items():
            scores[role] = agent.can_handle(query)

        # Select primary agent
        primary = max(scores, key=scores.get)

        # Determine if multi-agent collaboration needed
        high_scoring = [r for r, s in scores.items() if s > 0.3]
        is_complex = len(high_scoring) > 1 or len(query.split()) > 50

        # Check if RAG context is needed
        rag_keywords = ['our', 'company', 'policy', 'standard', 'guideline',
                        'process', 'internal', 'team', 'project']
        requires_rag = any(kw in query.lower() for kw in rag_keywords)

        steps = [{'step': 1, 'agent': primary.value, 'action': 'primary_task'}]
        if is_complex:
            for role in high_scoring:
                if role != primary:
                    steps.append({
                        'step': len(steps) + 1,
                        'agent': role.value,
                        'action': 'support_task'
                    })

        if requires_rag:
            steps.insert(0, {'step': 0, 'agent': 'rag_engine', 'action': 'context_retrieval'})

        plan = TaskPlan(
            task_id=f"task_{len(self.task_log) + 1:04d}",
            original_query=query,
            steps=steps,
            assigned_agents=[s['agent'] for s in steps],
            requires_rag=requires_rag,
            estimated_complexity='high' if is_complex else 'medium' if requires_rag else 'low'
        )

        self.task_log.append(plan)
        return plan

    def execute(self, query: str, rag_context: List[str] = None) -> Dict:
        """Execute a user query through the agent pipeline."""
        logger.info(f"\n{'='*60}")
        logger.info(f"ORCHESTRATOR: Processing query")
        logger.info(f"{'='*60}")

        # Plan
        plan = self.plan_task(query)
        logger.info(f"Task ID: {plan.task_id}")
        logger.info(f"Complexity: {plan.estimated_complexity}")
        logger.info(f"Agents: {plan.assigned_agents}")
        logger.info(f"RAG Required: {plan.requires_rag}")

        # Build context
        context = {
            'original_query': query,
            'task_plan': plan.__dict__,
        }
        if rag_context:
            context['retrieved_knowledge'] = rag_context

        # Execute through agents
        results = []
        for step in plan.steps:
            agent_name = step['agent']
            if agent_name == 'rag_engine':
                continue

            for role, agent in self.agents.items():
                if role.value == agent_name:
                    result = agent.process(query, context)
                    results.append(result)
                    break

        # Compile final response
        response = {
            'task_id': plan.task_id,
            'query': query,
            'plan': plan.__dict__,
            'agent_results': results,
            'status': 'completed',
            'rag_context_used': bool(rag_context),
        }

        logger.info(f"✅ Task {plan.task_id} completed | Agents: {len(results)}")
        return response


if __name__ == '__main__':
    orchestrator = MasterOrchestrator()

    queries = [
        "Write a Python function to process CSV files and generate summary statistics",
        "Draft a compliance report based on our data privacy policy",
        "Analyze our Q3 sales data and identify trends",
        "Research the latest transformer architectures for NLP",
        "Review this code and suggest improvements for performance",
    ]

    for q in queries:
        result = orchestrator.execute(q)
        print(f"\n📋 Task: {result['task_id']}")
        print(f"   Query: {q[:60]}...")
        print(f"   Agents: {[r['agent'] for r in result['agent_results']]}")
        print(f"   Status: {result['status']}")
