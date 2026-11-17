
import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document


class ChromaVectorStore:
    """ChromaDB vector store using Ollama embeddings."""

    def __init__(
        self,
        embeddings,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "financial_reports"
    ) -> None:

        self.client = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )

    def upload_chunks(
        self,
        chunks,
        embeddings,
        company: str,
        year: str,
        source_file: str
    ) -> None:
        """
        Upload chunks to ChromaDB.
        """

        documents = []
        ids = []

        for chunk in chunks:

            documents.append(
                Document(
                    page_content=chunk.page_content,
                    metadata={
                        "company": company,
                        "year": str(year),
                        "source_file": source_file
                    }
                )
            )

            ids.append(str(uuid.uuid4()))

        self.client.add_documents(
            documents=documents,
            ids=ids
        )

        print(
            f"Uploaded {len(documents)}/{len(chunks)} chunks to ChromaDB."
        )


class Retriever:
    """Retrieve relevant chunks from ChromaDB."""

    def __init__(self, client):
        self.client = client

    def invoke(
        self,
        query: str,
        company: str | list[str] | None = None,
        year: int | None = None,
        top_k: int = 8
    ) -> list:
        """
        Retrieve relevant chunks from ChromaDB.

        Returns a list of Document objects with page_content.
        """

        filters = []

        if isinstance(company, list) and company:
            filters.append({"company": {"$in": company}})
        elif company:
            filters.append({
                "company": company
            })

        if year is not None:
            filters.append({
                "year": str(year)
            })

        filter_expr = None

        if len(filters) == 1:
            filter_expr = filters[0]

        elif len(filters) > 1:
            filter_expr = {
                "$and": filters
            }

        documents = self.client.similarity_search(
            query=query,
            k=top_k,
            filter=filter_expr
        )

        return documents
