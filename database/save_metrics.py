from sqlalchemy import text

from database.sqlite_db import get_engine, initialize_database


def _value(metrics: dict, *keys: str):
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return "\n".join(str(item) for item in value) if isinstance(value, list) else str(value)
    return None


def save_metrics(company: str, year: int | None, metrics: dict) -> None:
    initialize_database()
    query = text("""INSERT INTO financial_report
        (company, year, revenue, net_income, operating_income, cash_flow, total_assets,
         total_liabilities, risk_factors, growth_drivers)
        VALUES (:company, :year, :revenue, :net_income, :operating_income, :cash_flow,
         :total_assets, :total_liabilities, :risk_factors, :growth_drivers)""")
    params = {
        "company": company, "year": str(year) if year is not None else None,
        "revenue": _value(metrics, "Revenue", "revenue"),
        "net_income": _value(metrics, "Net Income", "net_income"),
        "operating_income": _value(metrics, "Operating Income", "operating_income"),
        "cash_flow": _value(metrics, "Cash Flow from Operating Activities", "cash_flow"),
        "total_assets": _value(metrics, "Total Assets", "total_assets"),
        "total_liabilities": _value(metrics, "Total Liabilities", "total_liabilities"),
        "risk_factors": _value(metrics, "Top Risk Factors", "risk_factors"),
        "growth_drivers": _value(metrics, "Top Growth Drivers", "growth_drivers"),
    }
    with get_engine().begin() as connection:
        connection.execute(query, params)
