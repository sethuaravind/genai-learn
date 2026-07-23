# Chunking vs Tokenization

Chunking vs Tokenization: two words people use interchangeably. They shouldn't.

Both split text. That's where the similarity ends.

## Tokenization — for the model

- Breaks text into sub-word units (tokens) the model can actually process
- Uses algorithms like **BPE**, **WordPiece**, or **SentencePiece**
- Vocabulary is fixed at training time (e.g., ~100K tokens for GPT-style models)
- Happens on every inference call, invisible to the user
- Goal: convert language into numbers the transformer can compute on

## Chunking — for retrieval

- Breaks documents into passages before embedding, for a RAG pipeline
- Strategy is a design decision you control: fixed-size, sentence-based, semantic, recursive
- Chunk size is measured in tokens, but the split is driven by meaning/structure, not vocabulary
- Happens once, at ingestion time, when building your vector index
- Goal: create retrievable units that preserve enough context to be useful when retrieved

> Here's the part that trips people up: chunking decisions are made in terms of tokens. When you set `chunk_size=512`, you're counting tokens. So chunking depends on tokenization, but it solves a completely different problem.

- Get tokenization wrong → the model misreads your input.
- Get chunking wrong → the model never sees the right passage in the first place.

In RAG systems, I've seen more retrieval failures traced back to bad chunking (splitting mid-thought, losing headers/context, arbitrary fixed windows) than to tokenization issues. Tokenization is largely out of your hands. Chunking is entirely in your hands — and it's usually the highest-leverage lever in the whole pipeline.

If your RAG answers feel “almost right but missing context,” check your chunking strategy before you touch anything else.