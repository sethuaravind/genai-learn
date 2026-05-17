import os
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def read_pdf_text(pdf_path: str) -> str:
    """Read the full text of a PDF file using PyMuPDF."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    document = fitz.open(pdf_path)
    text_parts = []

    for page in document:
        text_parts.append(page.get_text())

    document.close()
    return "\n".join(text_parts)


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Split text into chunks using LangChain's RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)


def get_openai_embeddings(model_name: str = "text-embedding-3-large") -> OpenAIEmbeddings:
    """Return an OpenAI embeddings client for the given model."""
    return OpenAIEmbeddings(model=model_name)


def build_faiss_store(chunks: list[str], embeddings: OpenAIEmbeddings, persist_directory: str) -> FAISS:
    """Build and persist a local FAISS vector store from text chunks."""
    os.makedirs(persist_directory, exist_ok=True)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore.save_local(persist_directory)
    return vectorstore