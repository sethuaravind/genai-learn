# LLM Research Papers Bot

A research paper question-answering system built with PDF ingestion, Pinecone vector store indexing, and OpenAI RetrievalQA.

## Features
- Load research PDFs from `data/`
- Split documents into chunks for retrieval
- Create or connect to a Pinecone vector store
- Perform similarity search over research documents
- Answer user queries using LLM base QA Retrieval

## Prerequisites
- Python 3.10+ recommended
- `pip install -r requirements.txt`
- A valid OpenAI API key
- A valid Pinecone API key

## Setup
1. Place your PDF research papers in `RAG/LLMResearchPapersBot/data/`.
2. Create a `.env` file in the `RAG/LLMResearchPapersBot/` folder:

```
OPENAI_API_KEY=sk-xxxx
PINECONE_API_KEY=pcsk_xxxx
```

3. Confirm your Pinecone configuration in `RAG/LLMResearchPapersBot/config.yaml`:

```yaml
Pinecone:
  index_name: "llm-research-papers-bot"
  namespace: "llm-papers"
```

## Example research papers
- LLMOps - https://ieeexplore.ieee.org/abstract/document/10612341
- RAG - https://www.sciencedirect.com/science/article/pii/S1877050924021860
- LLM Agents - https://www.mdpi.com/2078-2489/16/2/87

Download these PDFs and save them under `RAG/LLMResearchPapersBot/data/`.

## Running the bot
- Open `RAG/LLMResearchPapersBot/main.py`.
- Set `upload = True` if you need to upload the local PDF documents into Pinecone.
- Run:

```bash
python main.py
```

The script will:
- load PDF files from `data/`
- split them into chunks
- create or connect to the Pinecone vector store
- run a similarity search
- answer the example query using the RetrievalQA chain

## Vector store QA usage
The LLM model is designed to accept a Pinecone vector store as input.
Use `llm_model.py` helpers like:

```python
from llm_model import answer_question_from_vectorstore

answer = answer_question_from_vectorstore(vector_store, "What are the main findings?", k=3)
print(answer["answer"])
```

This function builds a retrieval chain from the provided vector store and returns:
- `question`
- `answer`
- `raw`

## Notes
- Keep your API keys secure.
- If `upload` is `False`, `main.py` will attempt to connect to an existing Pinecone index.
- The default LLM model is `gpt-5.4-mini-2026-03-17` and embeddings use `text-embedding-3-large`.

