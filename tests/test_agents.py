"""Tests for AI agent modules."""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.code_agent import CodeAgent, CodeTask
from src.agents.data_agent import DataAgent
from src.agents.document_agent import DocumentAgent, DocumentRequest, DocumentType
import numpy as np
import pandas as pd


class TestCodeAgent:
    def setup_method(self):
        self.agent = CodeAgent()

    def test_generate_returns_result(self):
        task = CodeTask('t1', 'Write a sorting function', 'python')
        result = self.agent.generate(task)
        assert result.code
        assert result.task_id == 't1'

    def test_syntax_check_valid_code(self):
        assert self.agent._run_syntax_check('x = 1 + 1', 'python')

    def test_syntax_check_invalid_code(self):
        assert not self.agent._run_syntax_check('def bad(:\n  pass', 'python')

    def test_review_suggests_docstrings(self):
        code = 'def my_func(x):\n    return x * 2'
        notes = self.agent.review(code)
        assert any('docstring' in n.lower() for n in notes)

    def test_stats_tracks_tasks(self):
        for i in range(3):
            self.agent.generate(CodeTask(f't{i}', f'Task {i}', 'python'))
        stats = self.agent.stats()
        assert stats['tasks_completed'] == 3


class TestDataAgent:
    def setup_method(self):
        self.agent = DataAgent()
        self.df = pd.DataFrame({
            'revenue': np.random.lognormal(10, 1, 200),
            'units': np.random.randint(1, 100, 200),
            'region': np.random.choice(['A', 'B', 'C'], 200),
        })

    def test_analyze_returns_result(self):
        result = self.agent.analyze(self.df, 'test_data')
        assert result.dataset_name == 'test_data'
        assert result.shape == (200, 3)

    def test_insights_are_generated(self):
        result = self.agent.analyze(self.df)
        assert len(result.key_insights) > 0

    def test_report_contains_sections(self):
        result = self.agent.analyze(self.df)
        report = self.agent.generate_report(result)
        assert 'Key Insights' in report
        assert 'Recommendations' in report

    def test_missing_data_detected(self):
        df_missing = self.df.copy()
        df_missing.loc[:50, 'revenue'] = None
        result = self.agent.analyze(df_missing)
        assert any('missing' in i.lower() for i in result.key_insights)


class TestDocumentAgent:
    def setup_method(self):
        self.agent = DocumentAgent()

    def test_draft_email(self):
        req = DocumentRequest(DocumentType.EMAIL, 'Budget Update', audience='Finance')
        doc = self.agent.draft(req)
        assert doc.word_count > 10
        assert 'Budget Update' in doc.content

    def test_draft_report(self):
        req = DocumentRequest(DocumentType.REPORT, 'Q4 Analysis', audience='Executives')
        doc = self.agent.draft(req)
        assert 'Q4 Analysis' in doc.content
        assert doc.word_count > 50

    def test_stats_tracks_documents(self):
        for dt in [DocumentType.EMAIL, DocumentType.REPORT, DocumentType.MEETING_SUMMARY]:
            self.agent.draft(DocumentRequest(dt, 'Test Topic'))
        stats = self.agent.stats()
        assert stats['documents_drafted'] == 3
