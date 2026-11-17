
import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from ingestion.pdf_to_markdown import PDFToMarkdownConverter
from ingestion.semantic_chunker import create_semantic_splitter, read_markdown


load_dotenv()


# Local ChromaDB configuration
CHROMA_DB_DIR = "data/chroma_db"
COLLECTION_NAME = "financial_reports"
INGEST_LOCK = threading.Lock()


def _root_path(path: str) -> str:
    candidate = Path(path)
    return str(candidate if candidate.is_absolute() else Path(__file__).resolve().parents[1] / candidate)


def parse_company_year(pdf_file: Path) -> tuple[str, str]:
    """Parse company and year from a PDF filename."""

    stem = pdf_file.stem
    parts = stem.split("_")

    if len(parts) >= 2 and parts[0].isdigit():
        year = parts[0]
        company = " ".join(parts[1:])

    elif len(parts) >= 2:
        company = parts[0]
        year = parts[1]

    else:
        company = stem
        year = ""

    return company, year


def create_embeddings():
    """Initialize local Ollama embedding model."""

    return OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )


def create_vector_store(embeddings):
    """Initialize persistent local ChromaDB."""

    db_dir = _root_path(os.getenv("CHROMA_DB_DIR", CHROMA_DB_DIR))
    Path(db_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=db_dir
    )


def ingest_document(
    pdf_path: str,
    embeddings,
    vector_store,
    original_filename: str | None = None,
    progress_callback=None,
) -> dict:
    """
    Ingest a single PDF document.
    """

    pdf_file = Path(pdf_path)

    display_file = Path(original_filename).name if original_filename else pdf_file.name
    company, year = parse_company_year(Path(display_file))

    print(
        f"Ingesting {pdf_file.name} "
        f"as company={company!r}, year={year!r}"
    )

    if progress_callback:
        progress_callback("Converting PDF to text")

    # Step 1: Convert PDF into Markdown
    converter = PDFToMarkdownConverter()

    markdown_file = converter.convert_pdf(
        pdf_path=pdf_path,
        output_dir=_root_path("data/markdown")
    )

    print(f"Markdown generated: {markdown_file}")

    # SemanticChunker embeds sentences for boundaries; Chroma embeds resulting
    # chunks separately because those vectors are used for search.
    if progress_callback:
        progress_callback("Creating semantic chunks and embeddings")
    markdown_content = read_markdown(markdown_file)
    splitter = create_semantic_splitter(embeddings)
    chunks = splitter.create_documents([markdown_content])
    if not chunks:
        raise ValueError(f"No readable text was extracted from {display_file}")

    print(
        f"Generated {len(chunks)} chunks "
        f"for {pdf_file.name}"
    )

    # Step 3: Attach metadata to every chunk
    for index, chunk in enumerate(chunks):

        chunk.metadata.update({
            "company": company,
            "year": year,
            "source_file": display_file,
            "chunk_index": index
        })

    # Step 4: Store chunks and embeddings in ChromaDB
    if progress_callback:
        progress_callback(f"Saving {len(chunks)} chunks to ChromaDB")
    existing = vector_store.get(where={"source_file": display_file})
    if existing.get("ids"):
        vector_store.delete(ids=existing["ids"])
    vector_store.add_documents(
        documents=chunks,
        ids=[f"{display_file}:{index}" for index in range(len(chunks))],
    )

    print(f"Successfully stored chunks for {pdf_file.name}")

    print(f"Indexed {display_file}; KPI extraction can be run separately.")
    return {"company": company, "year": year, "chunks": len(chunks)}


def ingest_directory(input_dir: str) -> None:
    """
    Ingest all PDF documents from a directory.
    """

    # Initialize Ollama embeddings once
    embeddings = create_embeddings()

    # Initialize ChromaDB once
    vector_store = create_vector_store(embeddings)

    pdf_files = list(
        Path(input_dir).glob("*.pdf")
    )

    print(f"Found {len(pdf_files)} PDF(s)")

    for pdf_file in pdf_files:

        ingest_document(
            pdf_path=str(pdf_file),
            embeddings=embeddings,
            vector_store=vector_store
        )


if __name__ == "__main__":

    ingest_directory("data/raw_pdfs")
