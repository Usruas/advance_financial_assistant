from sqlalchemy import text

from database.sqlite_db import get_engine, initialize_database


def get_metrics():
    initialize_database()
    query = text("""SELECT id, company, year, revenue, net_income, operating_income,
        cash_flow, total_assets, total_liabilities, risk_factors, growth_drivers, created_at
        FROM financial_report ORDER BY company, year DESC, created_at DESC""")
    with get_engine().connect() as connection:
        return [dict(row._mapping) for row in connection.execute(query)]
