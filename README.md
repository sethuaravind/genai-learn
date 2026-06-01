# genai-learn

Learning notes and reference implementations for Generative AI (GenAI) topics.

Overview
- Short tutorials and explanations for core concepts used in LLM systems.
- Minimal, runnable examples demonstrating Retrieval-Augmented Generation (RAG) workflows.

## Concepts
- transformers
- Tokenization
- Prompt Engineering
- Chunking

## Hands-on
RAG
    - RAG/FDA_Guidelines — example RAG implementation that indexes an FDA guidance PDF and answers questions using a retrieval + LLM chain.

Quick start
1. Create and activate a Python virtual environment (recommended):

```bash
python -m venv .venv
.
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Populate the vectorstore with your documents (see `RAG/FDA_Guidelines/utils.py` for helpers) and run queries via the helper in `RAG/FDA_Guidelines/llm.py`.
