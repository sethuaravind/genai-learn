# Stack Overflow RAG Project - Reranking

This project demonstrates a retrieval-augmented generation (RAG) workflow using Stack Overflow data. It shows how to:

- prepare Stack Overflow CSV data for indexing
- build a Pinecone vector store with OpenAI embeddings
- retrieve top-k candidates from the vector store
- Rerank results to improve retrieval quality

## Project structure

- `data/` — raw and prepared Stack Overflow CSV files
- `Data_Preparation.ipynb` — notebook for converting CSV rows into documents
- `vectorstore.py` — script for creating and uploading documents to Pinecone
- `reranking.py` — script for retrieving results and applying reranking
- `reranked_results.txt` — output file for reranked retrieval results
- `extraction_results_without_reranking.txt` — output file for raw retrieval results

## Setup

1. Download the Stack Overflow dataset and place it in `stackoverflow_reranking/data/`.
   - Example source: https://www.kaggle.com/datasets/stackoverflow/stacksample
2. Create a `.env` file in the `stackoverflow_reranking/` folder.
3. Add your API keys and index settings:

```env
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
PINECONE_NAMESPACE=
COHERE_API_KEY=
```

## Usage

### 1) Prepare documents

Open and run `Data_Preparation.ipynb` to convert CSV rows into LangChain documents.

### 2) Build the vector store

Run:

```bash
python vectorstore.py --csv-path RAG/stackoverflow_reranking/data/formatted_stack_data.csv
```

Optional arguments:

- `--index-name` — Pinecone index name
- `--namespace` — Pinecone namespace
- `--max-rows` — limit the number of rows for testing
- `--batch-size` — batch size when uploading documents

### 3) Retrieve and rerank

Run:

```bash
python reranking.py
```

This will generate:

- `extraction_results_without_reranking.txt` — raw retrieval output
- `reranked_results.txt` — reranked retrieval output

## Notes

- Ensure your `.env` file is loaded before running scripts.
- The project currently uses Pinecone for the vector store and OpenAI for embeddings.
- Reranking is optional but can help improve relevance for the top candidates.

## Status

This project is in progress and may require updates to the reranking workflow or model configuration.