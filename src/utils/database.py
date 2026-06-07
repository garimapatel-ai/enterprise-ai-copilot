"""
Database utilities — PostgreSQL connection, session management, and schema definitions.
"""

import logging
from typing import Generator, Optional
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    doc_id: str
    title: str
    source: str
    team: str
    created_at: str
    sensitivity: str   # public, internal, confidential, restricted
    chunk_count: int


@dataclass
class ConversationRecord:
    session_id: str
    user_id: str
    messages: list
    agent_used: str
    created_at: str
    tokens_used: int
    cost_usd: float


class DatabaseManager:
    """
    Manages database connections and CRUD operations.
    In production: uses SQLAlchemy + PostgreSQL.
    This implementation uses in-memory storage for demonstration.
    """

    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or 'memory://'
        self._documents: dict = {}
        self._conversations: dict = {}
        self._users: dict = {}
        logger.info(f"Database initialized: {self.connection_string}")

    def save_document(self, record: DocumentRecord) -> bool:
        self._documents[record.doc_id] = record
        logger.debug(f"Saved document: {record.doc_id}")
        return True

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        return self._documents.get(doc_id)

    def list_documents(self, team: Optional[str] = None,
                       sensitivity: Optional[str] = None) -> list:
        docs = list(self._documents.values())
        if team:
            docs = [d for d in docs if d.team == team]
        if sensitivity:
            docs = [d for d in docs if d.sensitivity == sensitivity]
        return docs

    def save_conversation(self, record: ConversationRecord) -> bool:
        self._conversations[record.session_id] = record
        return True

    def get_conversation(self, session_id: str) -> Optional[ConversationRecord]:
        return self._conversations.get(session_id)

    def get_user_conversations(self, user_id: str) -> list:
        return [c for c in self._conversations.values() if c.user_id == user_id]

    def register_user(self, user_id: str, role: str, teams: list) -> bool:
        self._users[user_id] = {'role': role, 'teams': teams, 'created_at': datetime.now().isoformat()}
        return True

    def check_access(self, user_id: str, sensitivity: str) -> bool:
        """RBAC: check if user can access documents at given sensitivity level."""
        user = self._users.get(user_id, {})
        role = user.get('role', 'viewer')
        access_map = {
            'admin': ['public', 'internal', 'confidential', 'restricted'],
            'manager': ['public', 'internal', 'confidential'],
            'analyst': ['public', 'internal'],
            'viewer': ['public'],
        }
        return sensitivity in access_map.get(role, [])

    def stats(self) -> dict:
        return {
            'documents': len(self._documents),
            'conversations': len(self._conversations),
            'users': len(self._users),
        }


if __name__ == '__main__':
    db = DatabaseManager()

    db.register_user('user_001', 'analyst', ['engineering', 'data'])
    db.register_user('user_002', 'admin', ['all'])

    doc = DocumentRecord('doc_001', 'AI Strategy 2024', 'Confluence',
                         'engineering', datetime.now().isoformat(), 'internal', 12)
    db.save_document(doc)

    print(f"Access check (analyst→internal): {db.check_access('user_001', 'internal')}")
    print(f"Access check (analyst→restricted): {db.check_access('user_001', 'restricted')}")
    print(f"DB stats: {db.stats()}")
