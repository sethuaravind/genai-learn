import logging
import os
from typing import Any, Dict, Optional

from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

logger = logging.getLogger(__name__)


def get_openai_llm(
    model_name: str = "gpt-5.4-mini-2026-03-17",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    api_key: Optional[str] = None,
) -> OpenAI:
    """Create an OpenAI LLM client for answering research paper questions."""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    return OpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens, api_key=api_key)


def build_qa_chain(
    vectorstore: Any,
    llm: Optional[OpenAI] = None,
    model_name: str = "gpt-5.4-mini-2026-03-17",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    k: int = 4,
) -> Any:
    """Build a RetrievalQA chain from a vectorstore and an OpenAI LLM."""
    if llm is None:
        llm = get_openai_llm(model_name=model_name, temperature=temperature, max_tokens=max_tokens)

    if hasattr(vectorstore, "as_retriever"):
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    else:
        raise TypeError("vectorstore must implement as_retriever(search_kwargs={...})")

    prompt = PromptTemplate(
        input_variables=["input", "context"],
        template=(
            "You are an expert assistant that answers questions using only the provided research paper context.\n"
            "If the answer is not present in the context, reply with: 'I could not find an answer in the provided documents.'\n\n"
            "Question: {input}\n\n"
            "Context:\n{context}\n\n"
            "Answer precisely using the provided context and avoid hallucinating new information."
        ),
    )

    documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, documents_chain)


def answer_question_from_vectorstore(
    vectorstore: Any,
    question: str,
    model_name: str = "gpt-5.4-mini-2026-03-17",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    k: int = 4,
) -> Dict[str, Any]:
    """Answer a user question using the provided vectorstore."""
    if not question or not question.strip():
        raise ValueError("A non-empty question must be provided.")

    chain = build_qa_chain(
        vectorstore=vectorstore,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        k=k,
    )

    result = chain.invoke({"input": question})
    logger.debug("Retrieval QA result returned")
    return {
        "question": question,
        "answer": result.get("answer"),
        "raw": result,
    }


if __name__ == "__main__":
    print("This module provides RetrievalQA helpers. Use build_qa_chain() or answer_question_from_vectorstore().")
