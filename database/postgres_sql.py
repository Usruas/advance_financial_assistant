"""Backward-compatible database module; KPI storage is local SQLite."""
from database.sqlite_db import get_engine

__all__ = ["get_engine"]
