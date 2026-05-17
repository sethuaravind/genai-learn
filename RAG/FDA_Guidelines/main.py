import os
from dotenv import load_dotenv
from RAG.FDA_Guidelines.utils import *

load_dotenv()





if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    pdf_file = os.path.join(base_dir, "data", "FDA_Guidelines.pdf")
    persist_dir = os.path.join(base_dir, "faiss_store")
    text = read_pdf_text(pdf_file)
    chunks = split_text(text)
    embeddings = get_openai_embeddings()
    vectorstore = build_faiss_store(chunks, embeddings, persist_dir)
    print(f"Saved FAISS vector store to {persist_dir}")
    print(f"Index contains {vectorstore.index.ntotal if hasattr(vectorstore.index, 'ntotal') else 'unknown'} vectors")
