# Chunking Strategies
Chunking is not a preprocessing detail — it is the single most impactful decision in RAG pipeline quality. Poor chunking causes:

Too large  →  Retrieved chunk has relevant + irrelevant content
             →  Model gets confused, answer quality drops

Too small  →  Retrieved chunk lacks enough context
             →  Model gives incomplete answers

Wrong boundary →  Sentence split mid-thought
               →  Semantic meaning destroyed


The goal: every chunk should be semantically self-contained — a reader should understand it without needing surrounding context.

## Overview
Chunking Strategies
│
├── Fixed-Size Chunking          ← baseline, naive
├── Sentence-Based Chunking      ← better boundaries
├── Recursive Chunking           ← LangChain default
├── Semantic Chunking            ← meaning-aware splits
├── Hierarchical Chunking        ← structure-aware, multi-level
└── Late Chunking                ← newest, embedding-first

## 1. Fixed-Size Chunking
Split document every N tokens/characters regardless of content.

Cons
- Destroys semantic meaning
- Mixes unrelated sections
- No guarantee of capturing split context

## 2. Recursive Chunking
LangChain's default — splits on a hierarchy of separators, trying each until chunks are small enough.

Better than fixed-size — respects natural text boundaries. But still structure-agnostic.

## 3. Semantic Chunking
Uses embedding similarity to find natural breakpoints — splits where meaning changes significantly.
Core idea: consecutive sentences with high similarity belong together. When similarity drops sharply, that's a chunk boundary.

## 4. Hierarchical Chunking
Stores document at multiple levels of granularity simultaneously — big chunks for context, small chunks for precision. Retrieval uses small chunks but returns their parent for context.