import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


def load_environment() -> None:
    """Load environment variables from the local .env file when present."""
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()


def resolve_value(row: Dict[str, Any], *keys: str) -> Optional[str]:
    """Return the first non-empty value from the provided column names."""
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
        if value:
            return str(value)
    return None


def build_document(row: Dict[str, Any], row_number: int, source_path: str) -> Document:
    """Create a LangChain Document from a CSV row."""
    title = resolve_value(row, "title", "Title", "question_title", "Question title")
    body = resolve_value(
        row,
        "body",
        "Body",
        "question",
        "Question",
        "question_body",
        "Question body",
    )
    answers = resolve_value(row, "answers", "Answers", "answer", "Answer")

    parts: List[str] = []
    if title:
        parts.append(f"Title: {title}")
    if body:
        parts.append(f"Question: {body}")
    if answers:
        parts.append(f"Answers: {answers}")

    if not parts:
        parts = [value for value in row.values() if isinstance(value, str) and value.strip()]

    content = "\n\n".join(part for part in parts if part)

    metadata: Dict[str, Any] = {"source": source_path, "row_number": row_number}
    for key in ("id", "Id", "question_id", "QuestionId", "post_id", "PostId"):
        value = row.get(key)
        if value:
            metadata[key] = value

    return Document(page_content=content, metadata=metadata)


def read_csv_documents(csv_path: Path, max_rows: Optional[int] = None) -> List[Document]:
    """Read a CSV file and convert each row into a LangChain Document."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    documents: List[Document] = []
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {csv_path}")

        for idx, row in enumerate(reader, start=1):
            if max_rows is not None and idx > max_rows:
                break
            document = build_document(row=row, row_number=idx, source_path=str(csv_path))
            if document.page_content.strip():
                documents.append(document)

    if not documents:
        raise ValueError(f"No readable rows found in {csv_path}")

    return documents


def create_pinecone_vector_store(
    documents: List[Document],
    index_name: str,
    namespace: str = "",
    batch_size: int = 100,
) -> PineconeVectorStore:
    """Create or update a Pinecone vector store using OpenAI embeddings."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is not set")

    pinecone_client = Pinecone(api_key=pinecone_api_key)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)

    print(f"Uploading {len(documents)} documents to Pinecone index '{index_name}'...")
    vector_store = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace,
        batch_size=batch_size,
        pinecone_api_key=pinecone_api_key,
    )

    print(f"Upload complete for index '{index_name}'")
    return vector_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a CSV into a Pinecone vector store with OpenAI embeddings")
    parser.add_argument(
        "--csv-path",
        default=r"RAG\stackoverflow_reranking\data\formatted_stack_data.csv",
        help="Path to the CSV file to ingest",
    )
    parser.add_argument(
        "--index-name",
        default=os.getenv("PINECONE_INDEX_NAME", "stackoverflow-reranking"),
        help="Name of the Pinecone index to write into",
    )
    parser.add_argument(
        "--namespace",
        default=os.getenv("PINECONE_NAMESPACE", "stackoverflow"),
        help="Optional namespace for the Pinecone index",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for rows to upload for testing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of documents to send per batch",
    )
    return parser.parse_args()


def main() -> None:
    load_environment()
    args = parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    documents = read_csv_documents(csv_path=csv_path, max_rows=args.max_rows)

    create_pinecone_vector_store(
        documents=documents,
        index_name=args.index_name,
        namespace=args.namespace,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()