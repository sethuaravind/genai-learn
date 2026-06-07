# General RAG Flow for PDF Data

This directory demonstrates a basic Retrieval-Augmented Generation (RAG) pipeline using PDF documents as the source data.

## Workflow

1. Prepare source data
   - Add research PDFs to the `data/` folder.
2. Load PDF content
   - Read PDF files and extract text.
3. Split text into chunks
   - Break documents into smaller segments for better retrieval.
4. Create a vector store
   - Use FAISS or Pinecone to store embeddings.
5. Store embeddings
   - Convert text chunks into vector embeddings and save them.
6. Return the vector store
   - Provide the vector store to retrieval and QA logic.
7. Build a retrieval chain
   - Use document retrieval plus an LLM chain to answer questions.
8. Query with an LLM
   - Retrieve relevant context and generate answers from the model.

## Purpose

This RAG flow is intended to show how to:
- ingest PDF research material
- create searchable document embeddings
- connect a vector store to an LLM
- answer user queries using retrieved context

## Notes

- The repository includes an example implementation in `RAG/LLMResearchPapersBot/`.
- The example uses Pinecone as the production-ready vector store.
- You can replace Pinecone with a local FAISS index if preferred.

## Usage

1. Install dependencies from `requirements.txt`
2. Add your API keys to `.env`
3. Run the example script in `RAG/LLMResearchPapersBot/main.py`
4. Ask questions against the ingested PDF knowledge base
