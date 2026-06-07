"""
Prompt Templates — structured prompts for each agent type with few-shot examples.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    name: str
    system: str
    user_template: str
    few_shot_examples: List[Dict]
    output_format: str


TEMPLATES: Dict[str, PromptTemplate] = {

    'code_generation': PromptTemplate(
        name='code_generation',
        system="""You are an expert software engineer specializing in Python, ML systems, and APIs.
Generate production-ready code with:
- Type hints and docstrings
- Error handling and logging
- PEP 8 compliance
- Brief explanation of design choices""",
        user_template="""Task: {task_description}

Language: {language}
Context: {context}

Generate clean, well-documented code.""",
        few_shot_examples=[
            {
                'user': 'Write a FastAPI endpoint for user authentication',
                'assistant': '```python\n@app.post("/auth/login")\nasync def login(credentials: LoginModel):\n    """Authenticate user and return JWT token."""\n    user = await authenticate(credentials)\n    if not user:\n        raise HTTPException(401, "Invalid credentials")\n    return {"token": create_jwt(user.id)}\n```'
            }
        ],
        output_format='```python\n{code}\n```\n\n**Explanation:** {explanation}'
    ),

    'rag_answer': PromptTemplate(
        name='rag_answer',
        system="""You are a helpful enterprise AI assistant. Answer questions using ONLY the provided context.
If the context doesn't contain enough information, say so clearly.
Always cite the source document when referencing specific facts.
Be concise, accurate, and professional.""",
        user_template="""Context from knowledge base:
{context}

Question: {question}

Answer based on the context above. If uncertain, state your confidence level.""",
        few_shot_examples=[
            {
                'user': 'What is our refund policy?\nContext: [Doc: policy.pdf] Refunds are processed within 5 business days.',
                'assistant': 'According to our policy document, refunds are processed within 5 business days. [Source: policy.pdf]'
            }
        ],
        output_format='{answer}\n\nSources: {sources}'
    ),

    'data_analysis': PromptTemplate(
        name='data_analysis',
        system="""You are a senior data scientist. Analyze datasets and produce actionable insights.
Focus on: statistical significance, business impact, anomalies, and recommendations.
Use clear language accessible to both technical and non-technical stakeholders.""",
        user_template="""Dataset: {dataset_name}
Shape: {shape}
Summary Statistics:
{stats}

Task: {analysis_task}

Provide insights and recommendations.""",
        few_shot_examples=[],
        output_format='## Key Findings\n{findings}\n\n## Recommendations\n{recommendations}'
    ),

    'document_draft': PromptTemplate(
        name='document_draft',
        system="""You are an expert business writer. Draft clear, professional documents.
Adapt tone and length to the audience. Use structured formatting.
For technical audiences: include details. For executives: lead with impact.""",
        user_template="""Document Type: {doc_type}
Topic: {topic}
Audience: {audience}
Tone: {tone}
Key Points to Include:
{key_points}

Draft the document.""",
        few_shot_examples=[],
        output_format='{document_content}'
    ),

    'safety_check': PromptTemplate(
        name='safety_check',
        system="""You are a content safety classifier. Evaluate if responses are safe for enterprise use.
Flag: harmful content, PII exposure, proprietary data leaks, policy violations, hallucinations.
Return a JSON safety assessment.""",
        user_template="""Response to evaluate:
{response}

Check for: {check_types}

Return safety assessment.""",
        few_shot_examples=[],
        output_format='{{"safe": true/false, "issues": [], "confidence": 0.0-1.0}}'
    ),
}


def build_prompt(template_name: str, **kwargs) -> Dict[str, str]:
    """Build a prompt dict from a template and variables."""
    if template_name not in TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(TEMPLATES.keys())}")

    tmpl = TEMPLATES[template_name]
    try:
        user_content = tmpl.user_template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing template variable: {e}")

    messages = [{'role': 'system', 'content': tmpl.system}]
    for ex in tmpl.few_shot_examples:
        messages.append({'role': 'user', 'content': ex['user']})
        messages.append({'role': 'assistant', 'content': ex['assistant']})
    messages.append({'role': 'user', 'content': user_content})

    return {
        'messages': messages,
        'output_format': tmpl.output_format,
        'template_name': template_name,
    }


if __name__ == '__main__':
    prompt = build_prompt('code_generation',
                          task_description='Build a data validation pipeline',
                          language='Python',
                          context='Used in ETL before loading to PostgreSQL')
    print(f"Template: {prompt['template_name']}")
    print(f"Messages: {len(prompt['messages'])}")
    for m in prompt['messages']:
        print(f"  [{m['role']}]: {m['content'][:80]}...")
