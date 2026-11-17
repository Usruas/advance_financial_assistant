
from pathlib import Path

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings


def create_semantic_splitter(embeddings):
    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        buffer_size=1,
    )


def read_markdown(markdown_file: str) -> str:
    """
    Read markdown content.
    """
    return Path(markdown_file).read_text(encoding="utf-8")


def chunk_markdown(
    markdown_file: str,
    embeddings
) -> list[Document]:
    """
    Generate semantic chunks from markdown.
    """

    markdown_content = read_markdown(markdown_file)

    splitter = create_semantic_splitter(embeddings)

    return splitter.create_documents([markdown_content])


if __name__ == "__main__":

    # Initialize local Ollama embedding model
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    markdown_file = "../data/markdown/2024_Apple.md"

    chunks = chunk_markdown(
        markdown_file=markdown_file,
        embeddings=embeddings
    )

    print(f"Generated {len(chunks)} chunks\n")

    for index, chunk in enumerate(chunks[:3]):
        print("=" * 80)
        print(f"Chunk {index + 1}") 
        print("=" * 80)
        print(chunk.page_content[:1000])
        print()
