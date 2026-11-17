
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

from vectore_store.chromadb_Search import Retriever

load_dotenv()


class FinancialMetrics(BaseModel):
    revenue: str | int | None = Field(None, alias="Revenue")
    net_income: str | int | None = Field(None, alias="Net Income")
    operating_income: str | int | None = Field(None, alias="Operating Income")
    cash_flow: str | int | None = Field(
        None, alias="Cash Flow from Operating Activities"
    )
    total_assets: str | int | None = Field(None, alias="Total Assets")
    total_liabilities: str | int | None = Field(None, alias="Total Liabilities")
    risk_factors: str | list | None = Field(None, alias="Top Risk Factors")
    growth_drivers: str | list | None = Field(None, alias="Top Growth Drivers")


def retrieve_context(
    retriever: Retriever,
    company: str,
    year: int | None
) -> str:
    """
    Retrieve broad financial context from ChromaDB.
    """

    query = f"""
    Annual report financial statements,
    income statement,
    balance sheet,
    cash flow statement,
    risks,
    growth drivers,
    financial performance
    for {company} fiscal year {year or 'available in the report'}
    """

    documents = retriever.invoke(query)
    return "\n\n".join(
        doc.page_content for doc in documents
    )


def build_extraction_prompt(
    company: str,
    year: int | None,
    context: str
) -> str:
    """
    Build KPI extraction prompt.
    """

    return f"""
You are an expert financial analyst.

Company: {company}
Year: {year or 'identify from the report'}

Context:
{context}

Extract the following information:

1. Revenue
2. Net Income
3. Operating Income
4. Cash Flow from Operating Activities
5. Total Assets
6. Total Liabilities
7. Top Risk Factors
8. Top Growth Drivers

Instructions:

- Use only the provided context.
- Return null if unavailable.
- Financial values must match the report exactly.
- Risk factors should be concise.
- Growth drivers should be concise.
"""


def extract_financial_metrics(
    retriever: Retriever,
    company: str,
    year: int | None
) -> dict:
    """
    Extract KPIs using local Ollama LLM and ChromaDB.
    """

    # Step 1: Retrieve relevant context
    context = retrieve_context(
        retriever=retriever,
        company=company,
        year=year
    )

    # Step 2: Build extraction prompt
    prompt = build_extraction_prompt(
        company=company,
        year=year,
        context=context
    )

    # Step 3: Initialize local Ollama LLM
    llm = ChatOllama(
        model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.1"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0
    )

    # Step 4: Generate structured output
    structured_llm = llm.with_structured_output(
        FinancialMetrics
    )

    metrics = structured_llm.invoke(prompt)

    # Step 5: Return dictionary using original field aliases
    return metrics.model_dump(by_alias=True)


def main() -> None:

    company = "Apple"
    year = 2024

    # Initialize local embeddings
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # Initialize ChromaDB
    vector_store = Chroma(
        collection_name="financial_reports",
        embedding_function=embeddings,
        persist_directory=str(Path(__file__).resolve().parents[1] / os.getenv("CHROMA_DB_DIR", "data/chroma_db"))
    )

    # Initialize retriever
    retriever = Retriever(
        vector_store
    )

    # Extract financial metrics
    results = extract_financial_metrics(
        retriever=retriever,
        company=company,
        year=year
    )

    print(f"\nExtracted KPIs for {company} {year}\n")

    for key, value in results.items():
        print(f"{key}:")
        print(value)
        print("-" * 80)

    # Save extracted metrics
    from database.save_metrics import save_metrics

    save_metrics(
        company=company,
        year=year,
        metrics=results
    )


if __name__ == "__main__":
    main()
