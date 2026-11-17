from sqlalchemy import text

from database.sqlite_db import get_engine, initialize_database


def get_history(conversation_id: str, limit: int = 12) -> list[dict[str, str]]:
    initialize_database()
    with get_engine().connect() as connection:
        rows = connection.execute(text("""SELECT role, content FROM chat_messages
            WHERE conversation_id = :conversation_id ORDER BY id DESC LIMIT :limit"""),
            {"conversation_id": conversation_id, "limit": limit}).mappings().all()
    return [dict(row) for row in reversed(rows)]


def list_messages(conversation_id: str, limit: int = 24) -> list[dict[str, str]]:
    initialize_database()
    with get_engine().connect() as connection:
        rows = connection.execute(text("""SELECT role, content FROM chat_messages
            WHERE conversation_id = :conversation_id ORDER BY id DESC LIMIT :limit"""),
            {"conversation_id": conversation_id, "limit": limit}).mappings().all()
    return [dict(row) for row in reversed(rows)]


def save_turn(conversation_id: str, question: str, answer: str) -> None:
    initialize_database()
    with get_engine().begin() as connection:
        connection.execute(text("""INSERT INTO chat_messages (conversation_id, role, content)
            VALUES (:conversation_id, 'user', :question), (:conversation_id, 'assistant', :answer)"""),
            {"conversation_id": conversation_id, "question": question, "answer": answer})
        connection.execute(text("""DELETE FROM chat_messages WHERE conversation_id = :conversation_id
            AND id NOT IN (SELECT id FROM chat_messages WHERE conversation_id = :conversation_id
            ORDER BY id DESC LIMIT 24)"""), {"conversation_id": conversation_id})
