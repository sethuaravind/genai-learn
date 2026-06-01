# FDA Guidelines RAG

This module implements a small Retrieval-Augmented Generation (RAG) assistant for answering questions using FDA guideline documents.

**Purpose**
- Provide precise answers grounded in FDA guideline text using a vectorstore + LLM retrieval chain.

**Flow**
- Read PDF
- Chunk the document
- Embedd it
- Create a local vector store
- Create a document chain
- Create a document retrieval
- LLM chain
- Invoke the LLM

**Contents**
- `llm.py`: helpers to build the retrieval QA chain and answer questions programmatically. See [RAG/FDA_Guidelines/llm.py](RAG/FDA_Guidelines/llm.py#L1)
- `main.py`: (optional) small runner or example entrypoint.
- `utils.py`: utility helpers for data loading and formatting.

**Requirements**
- Python 3.10+
- Dependencies listed in the repository `requirements.txt` (install with `pip install -r requirements.txt`).

**Quick Start**
1. Ensure the vectorstore is populated with FDA documents (embeddings + vector index).
2. Create or obtain an LLM client and pass it to the chain builders.

Example usage (interactive or script):
```python
from RAG.FDA_Guidelines.llm import answer_question_from_fda

# `vectorstore` is your indexed store (FAISS, Milvus, etc.)
res = answer_question_from_fda(vectorstore, "What is the FDA guidance on X?")
print(res['answer'])
print(res['raw'])
```

**What `k` means**
- `k` is the top-k number of documents fetched from the vectorstore for each query (how many context candidates the retriever returns). Adjust `k` higher to include more context, or lower to reduce noise.

**Troubleshooting: chain returns empty or blank answer**
- Inspect the chain output keys. The code currently does `result.get("answer")` — confirm the chain actually returns an `answer` key:
```python
result = chain.invoke({"input": question})
print(result)
print(list(result.keys()))
```
- If the key is different (for example `output` or `text`), read that field instead.
- Check retriever results to ensure context exists:
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
docs = retriever.get_relevant_documents(question)
print(len(docs))
for d in docs:
	print(d.metadata.get('source'), d.page_content[:300])
```
- If `docs` is empty, increase `k`, verify your vector index is loaded, and ensure embeddings were created with the same model and consistent preprocessing.
- If `docs` contain text but chain output is empty, inspect the `documents_chain` prompt and LLM response logs (check for token limits or model errors).

**Testing**
- Run small interactive queries against known prompts and verify `answer` vs `raw` outputs.

**Next steps**
- Add example scripts in this folder to demonstrate end-to-end indexing → query flow.
