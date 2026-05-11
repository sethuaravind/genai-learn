# Tokenization

A token is not a word. It's a chunk of text — could be a word, subword, character, or even punctuation — that the model treats as its atomic unit.

| Approach | Problem |
| --- | --- |
| Word-level | Vocabulary explodes — "run", "running", "ran" = 3 entries. Unknown words crash the model |
| Character-level | Sequences become extremely long → attention complexity explodes. "hello" = 5 tokens |
| Subword (current) | Best of both — fixed vocab, handles unknown words, reasonable sequence length |

## 1. Byte-Pair Embedding
Training algorithm
Start with character vocabulary
Repeat:
  1. Count all adjacent pair frequencies
  2. Merge the most frequent pair into a new token
  3. Add merged token to vocabulary
Until vocabulary size reached

## 2. WordPiece — Used by BERT
Similar to BPE but merges based on likelihood rather than raw frequency:

score = freq(pair) / (freq(token1) × freq(token2))

## 3. SentencePiece + Unigram — Used by LLaMA, T5

Treats the input as a raw byte stream — no pre-tokenization on whitespace. Language-agnostic.

Uses a unigram language model — starts with a large vocabulary and prunes tokens that least affect corpus likelihood.