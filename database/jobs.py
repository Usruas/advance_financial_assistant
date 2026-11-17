from __future__ import annotations

from sqlalchemy import text

from database.sqlite_db import get_engine, initialize_database


def create_job(job_id: str, filename: str) -> None:
    initialize_database()
    with get_engine().begin() as connection:
        connection.execute(text("""INSERT INTO ingestion_jobs (job_id, filename, status, stage)
            VALUES (:job_id, :filename, 'queued', 'Waiting for worker')"""),
            {"job_id": job_id, "filename": filename})


def update_job(job_id: str, *, status: str | None = None, stage: str | None = None,
               error: str | None = None) -> None:
    updates = {key: value for key, value in {"status": status, "stage": stage,
               "error": error}.items() if value is not None}
    if not updates:
        return
    assignments = ", ".join(f"{key} = :{key}" for key in updates)
    updates["job_id"] = job_id
    with get_engine().begin() as connection:
        connection.execute(text(f"UPDATE ingestion_jobs SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE job_id = :job_id"), updates)


def get_job(job_id: str) -> dict | None:
    initialize_database()
    with get_engine().connect() as connection:
        row = connection.execute(text("SELECT job_id, filename, status, stage, error, created_at, updated_at FROM ingestion_jobs WHERE job_id = :job_id"), {"job_id": job_id}).first()
    return dict(row._mapping) if row else None
