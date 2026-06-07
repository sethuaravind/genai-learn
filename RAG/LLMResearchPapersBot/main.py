import yaml
from pathlib import Path
from dotenv import load_dotenv
from tools import process_pdfs_to_pinecone, get_pinecone_vector_store
from llm_model import answer_question_from_vectorstore


project_root = Path(__file__).resolve().parent
env_file = project_root / ".env"
research_papers_dir = project_root / "data"
config_file = project_root / "config.yaml"
load_dotenv(dotenv_path=env_file)


# Configuration
with open(config_file) as f:
    config = yaml.safe_load(f)
index_name = config["Pinecone"]["index_name"]
namespace = config["Pinecone"]["namespace"]

# Pinecone
upload = False

# Run the pipeline
if upload:
    vector_store = process_pdfs_to_pinecone(
        pdf_directory=research_papers_dir,
        index_name=index_name,
        namespace=namespace
    )

vector_store = get_pinecone_vector_store(index_name, namespace)

# Example: Search and answer with the vector store
query = "What are the main findings?"

# Similarity Search
results = vector_store.similarity_search(query, k=3)
print("\n=== Search Results ===")
for i, result in enumerate(results, 1):
    print(f"\n{i}. {result.metadata.get('source', 'Unknown')}")
    print(f"Content: {result.page_content[:200]}...")

# LLM RetrievalQA
print("\n=== Answer from LLM RetrievalQA ===")
answer = answer_question_from_vectorstore(vector_store, query, k=3)
print(answer.get("answer"))