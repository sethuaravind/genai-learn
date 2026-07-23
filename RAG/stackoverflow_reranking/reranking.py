import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

project_root = Path(__file__).resolve().parent
env_file = project_root / ".env"

load_dotenv(dotenv_path=env_file)

# variables
k = 50


vectorstore_index = os.getenv("PINECONE_INDEX_NAME")
vectorstore_namespace = os.getenv("PINECONE_NAMESPACE")


embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.getenv("OPENAI_API_KEY")
    )

vectorstore = PineconeVectorStore(
    index_name=vectorstore_index,
    embedding=embeddings,
    namespace=vectorstore_namespace
)

query = "Python not decoding json"

docs_with_scores = vectorstore.similarity_search_with_score(query, k)

output_file = project_root / "extraction_results_without_reranking.txt"
with output_file.open("w", encoding="utf-8") as f:
    for doc, score in docs_with_scores:
        f.write(f"{score}\t{doc.page_content}\n")


# Reranking
base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
compressor = CohereRerank(model="rerank-english-v3.0", top_n=5)


retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)


# use exactly like a normal retriever downstream
docs = retriever.invoke(query)

output_file = project_root / "reranked_results.txt"
with output_file.open("w", encoding="utf-8") as f:
    for i, doc in enumerate(docs):
        f.write(f"{i}\t{doc.page_content}\n----------------------\n")