# Different Ways to Call an LLM

This document shows common patterns for calling large language models (LLMs) used in this repository: retrieval-augmented generation (RAG) flows and direct text generation. Each example shows the minimal code and a short explanation.

---

## 1. Retrieval-Augmented Generation (RAG)

RAG combines a vector store retriever (to fetch relevant context) with an LLM to produce answers grounded in documents. Typical steps:

- Create or load a vectorstore and convert it to a retriever.
- Build a documents chain that formats retrieved context and the question for the LLM.
- Invoke the retrieval+generation chain to get an answer.

Example (pseudo-code):

```python
from langchain_openai import OpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# 1) Create LLM client
llm = OpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)

# 2) Build a documents chain (formats prompt + context)
documents_chain = create_stuff_documents_chain(llm, prompt)

# 3) Convert vectorstore to a retriever and create the retrieval chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
chain = create_retrieval_chain(retriever, documents_chain)

# 4) Invoke with the user's question
result = chain.invoke({"input": question})
print(result.get("answer"))
```

Notes:
- `search_kwargs={"k": ...}` controls how many documents are retrieved (top-k retrieval).
- The documents chain controls how retrieved text is combined and presented to the LLM.

---

## 2. Direct Text Generation (Chat/Completions)

Use this when you want the model to generate text directly from a prompt (no retrieval). This is suitable for open-ended generation, completion, or chat-style interactions.

Example using a chat-style API:

```python
from langchain_openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    max_tokens=50,
)
print(response.choices[0].message.content)
```

Key parameters:
- `temperature` (float): controls randomness. Lower -> more deterministic outputs.
- `top_p` (float): nucleus sampling; set to <1.0 to sample from the top cumulative probability mass.
- `max_tokens` (int): limits response length.

---

## When to use which

- Use **RAG** when you need answers grounded in your documents or knowledge base (e.g., research papers, product docs).
- Use **Direct generation** for open-ended creativity, chatbots without retrieval, or short completions.

If you want, I can add code examples for `top_k` / `top_p` sampling or a small runnable demo showing differences.