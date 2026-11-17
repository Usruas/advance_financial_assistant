from database.postgres_sql import get_engine
from sqlalchemy import text

engine = get_engine()

with engine.connect() as connection:
    result = connection.execute(text("SELECT version()"))
    print(result.scalar())

print("PostgreSQL connected successfully!")