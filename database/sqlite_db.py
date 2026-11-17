import os
from pathlib import Path

from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        configured = os.getenv("KPI_DATABASE_PATH", "data/financial_metrics.db")
        path = Path(configured)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{path.as_posix()}", connect_args={"check_same_thread": False})
    return _engine


def initialize_database():
    query = text("""CREATE TABLE IF NOT EXISTS financial_report (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company TEXT NOT NULL,
        year TEXT,
        revenue TEXT,
        net_income TEXT,
        operating_income TEXT,
        cash_flow TEXT,
        total_assets TEXT,
        total_liabilities TEXT,
        risk_factors TEXT,
        growth_drivers TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    with get_engine().begin() as connection:
        connection.execute(query)
        connection.execute(text("""CREATE TABLE IF NOT EXISTS ingestion_jobs (
            job_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            status TEXT NOT NULL,
            stage TEXT NOT NULL,
            error TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )"""))
        connection.execute(text("""CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )"""))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_chat_messages_conversation ON chat_messages(conversation_id, id)"))
