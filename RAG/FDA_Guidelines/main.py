import os
from dotenv import load_dotenv
from llm import answer_question_from_fda
from utils import *
from pathlib import Path

load_dotenv(Path().resolve() / ".env")

base_dir = os.path.dirname(__file__)
persist_dir = os.path.join(base_dir, "faiss_store")


def create_local_vectorstore():
    """Create and persist a local FAISS vectorstore from the FDA PDF."""
    pdf_file = os.path.join(base_dir, "data", "FDA_Guidelines.pdf")
    text = read_pdf_text(pdf_file)
    chunks = split_text(text)
    embeddings = get_openai_embeddings()
    vectorstore = build_faiss_store(chunks, embeddings, persist_dir)
    print(f"Vectorstore created and saved to {persist_dir}")
    return vectorstore


def get_vectorstore():
    """Load vectorstore from disk. Create if it doesn't exist."""
    if not os.path.exists(persist_dir):
        print(f"Vectorstore not found at {persist_dir}. Creating...")
        return create_local_vectorstore()
    else:
        embeddings = get_openai_embeddings()
        vectorstore = load_faiss_store(persist_dir, embeddings)
        print(f"Vectorstore loaded from {persist_dir}")
        return vectorstore


def chat(question: str) -> str:
    """Answer a question from the FDA guidelines document."""
    vectorstore = get_vectorstore()
    result = answer_question_from_fda(vectorstore, question)
    return result.get("answer", result)


if __name__ == "__main__":
    print(chat("Give a example for BIMO inspection in point wise format"))