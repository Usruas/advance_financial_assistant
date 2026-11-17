from sqlalchemy import text

from database.sqlite_db import initialize_database


def create_tables() -> None:
    initialize_database()
    print("SQLite financial_report table created.")


if __name__ == "__main__":
    create_tables()
