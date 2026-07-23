# Reranking in RAG Pipelines

## What It Is

Reranking is a second-stage retrieval step that reorders an initial set of candidate documents by relevance, using a more precise (but more expensive) model than the one used for the first retrieval pass.

In a typical RAG pipeline:

1. **Retriever** (bi-encoder / vector search) pulls top-k candidates (e.g., top 50–100) from a vector store like Pinecone or ChromaDB — fast, but approximate.
2. **Reranker** (cross-encoder) re-scores those candidates against the query with much higher precision, and you keep only the top-n (e.g., top 5–10) to pass into the LLM's context window.

## Why It's Needed

Vector similarity search (cosine/dot-product over embeddings) is fast because query and document embeddings are computed **independently** — that's what makes ANN indexes possible. But independence has a cost: the model never actually looks at the query and document *together*, so it misses fine-grained interactions (negation, term order, exact entity matches, multi-hop relevance).

A cross-encoder reranker takes `(query, document)` as a **joint input** and outputs a single relevance score. This is much more accurate but O(n) per query — it can't be pre-computed or indexed, so it's only feasible on a small candidate set, which is why it runs *after* the cheap retrieval step, not instead of it.

## Bi-Encoder vs Cross-Encoder

| | Bi-Encoder (Retriever) | Cross-Encoder (Reranker) |
|---|---|---|
| Input | Query and doc encoded separately | Query and doc encoded jointly |
| Output | Two embeddings → cosine similarity | Single relevance score |
| Speed | Fast (precomputed doc embeddings, ANN search) | Slow (no precomputation possible) |
| Scale | Millions of documents | Tens to hundreds of candidates |
| Precision | Lower — misses token-level interaction | Higher — full attention across query+doc |

## Common Reranker Choices

- **Cohere Rerank** (`rerank-english-v3.0` / multilingual) — hosted API, drop-in for most RAG stacks
- **BGE-reranker** (BAAI) — open-source, strong performance, self-hostable
- **ColBERT** — late-interaction model; a middle ground between bi- and cross-encoders (token-level embeddings with a cheap MaxSim operator, so it's more scalable than a full cross-encoder while still being more precise than single-vector similarity)
- **Cross-encoder MiniLM** (sentence-transformers) — lightweight, good for latency-sensitive setups

## Where It Fits in a LangChain-Style Pipeline

```
Query
  │
  ▼
Embed query (bi-encoder)
  │
  ▼
Vector search (Pinecone/ChromaDB) → top-k candidates (k~50-100)
  │
  ▼
Reranker (cross-encoder) scores each (query, doc) pair
  │
  ▼
Top-n candidates (n~5-10) → LLM context
  │
  ▼
Generation
```

In LangChain, this is typically wired via `ContextualCompressionRetriever` with a `CohereRerank` or custom cross-encoder compressor wrapping the base retriever.

## Practical Notes

- **Latency tradeoff**: reranking adds a network/inference hop. For latency-sensitive apps, cap candidate count aggressively (e.g., k=20 rather than k=100) before reranking.
- **Cost**: hosted rerankers (Cohere) charge per query+document pair — cost scales with k, not just query volume.
- **When it matters most**: reranking gives the biggest lift when your corpus has many *near-duplicate* or *topically similar but not actually relevant* documents — common in dense technical corpora (e.g., clinical trial protocols, engineering specs).
- **Diminishing returns**: if your retriever's top-5 is already highly precise (small, well-curated corpus), reranking adds latency without meaningfully improving answer quality — always A/B against your own retrieval quality metrics (recall@k, MRR) before adding it as a fixed pipeline stage.

## Evaluation

To justify adding a reranker, measure before/after on:
- **Recall@k** — did the right document make it into the final context at all?
- **MRR (Mean Reciprocal Rank)** — how high up did the right document rank?
- **Downstream answer accuracy** — does the LLM's final answer actually improve, since that's the metric that ultimately matters