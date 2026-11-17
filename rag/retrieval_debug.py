
import sys

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from vectore_store.chromadb_Search import Retriever


def search_vectorstore(query: str, top: int = 5):

    # Step 1: Initialize Ollama embeddings
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    # Step 2: Connect to existing ChromaDB
    store = Chroma(
        collection_name="financial_reports",
        embedding_function=embeddings,
        persist_directory="data/chroma_db"
    )

    # Step 3: Initialize Retriever
    retriever = Retriever(store)

    # Step 4: Perform similarity search
    results = retriever.invoke(
        query=query,
        top_k=top
    )

    print(f"Query: {query!r}")
    print(f"Top: {top}")
    print(f"Results: {len(results)}\n")

    # Step 5: Display results
    for idx, result in enumerate(results, start=1):

        content = result.page_content

        snippet = content.strip().replace("\n", " ")

        if len(snippet) > 350:
            snippet = snippet[:350].rstrip() + "..."

        print(f"Result {idx}")
        print(f"  Content snippet: {snippet}")
        print(f"  Metadata: {result.metadata}")
        print("  " + "-" * 60)

    if not results:
        print(
            "No results returned. Verify your ChromaDB collection "
            "contains documents or try another query."
        )

    return results


def main():

    if len(sys.argv) < 2:
        print(
            'Usage: python -m rag.retrieval_debug "your query here"'
        )
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    search_vectorstore(query)


if __name__ == "__main__":
    main()
