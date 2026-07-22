import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

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

for doc, score in docs_with_scores:
    print(score, doc.page_content[:100])