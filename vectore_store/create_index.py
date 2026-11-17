from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def create_index(
    embeddings,
    persist_directory: str = "data/chroma_db",
    collection_name: str = "financial_reports"
) -> None:
    """
    Create or initialize a local ChromaDB collection.

    Args:
        embeddings: Ollama embedding model.
        persist_directory: Directory where ChromaDB stores data.
        collection_name: Name of the collection.
    """

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    print(
        f"ChromaDB collection '{collection_name}' initialized successfully."
    )

    return vector_store


if __name__ == "__main__":

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    create_index(
        embeddings=embeddings,
        persist_directory="data/chroma_db",
        collection_name="financial_reports"
    )