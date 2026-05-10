# Transformers: The Architecture Behind Modern AI

Core Concepts: "For each word I'm processing, which other words in the sentence should I pay attention to?"

## Problem Statement
Before transformers (2017), RNNs and LSTMs processed text sequentially — word by word. This meant:
- Long-range dependencies were lost ("The cat that sat on the mat... was hungry" — was refers to cat, 10 words back)
- No parallelization → slow training
- Gradient vanishing on long sequences

## Key Components


### The Full Forward Pass (Simply)
- Input tokens
- Token Embeddings + Positional Encoding
- Repeat N times:
  - Multi-Head Self-Attention
  - Add & Norm
  - Feed-Forward Network
  - Add & Norm
- Final hidden states
- Linear + Softmax → Probability over vocabulary

### 1. Input Embeddings + Positional Encoding
- Words are converted to vectors (embeddings) — numbers that capture meaning
- Since transformers process everything in parallel (no sequence), they inject positional encoding to tell the model "this word is at position 3"
- Without this, "dog bites man" and "man bites dog" would look the same

### 2. Self-Attention (The Core Mechanism)
Every word creates 3 vectors:

Q (Query) — "What am I looking for?"
K (Key) — "What do I contain?"
V (Value) — "What do I actually pass forward?"

The attention score between two words = softmax(Q · Kᵀ / √d_k) × V
Intuition with an example:
"The animal didn't cross the street because it was too tired"
When processing "it", the model computes attention scores against every other word. "Animal" gets a high score → the model learns "it" refers to "animal", not "street".

### 3. Multi-Head Attention

Instead of doing attention once, transformers do it H times in parallel with different learned projections.
Why? Each head learns to attend to different relationships:

Head 1 → syntactic relationships (subject-verb)
Head 2 → coreference (it → animal)
Head 3 → positional proximity
Head 4 → semantic similarity

All heads concatenated → richer representation.

### 4. Feed-Forward Network (FFN)

After attention, each position passes through a simple 2-layer MLP independently:

FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

This is where the model stores factual knowledge — think of it as a key-value memory lookup. Research shows specific facts (e.g., "Paris is the capital of France") live in FFN weights.

### 5. Add & Norm (Residual Connections)
After every sub-layer:

output = LayerNorm(x + SubLayer(x))

- Residual connection (x +): prevents vanishing gradients, lets gradients flow directly
- Layer normalization: stabilizes training

This is why you can stack 96 layers in GPT-4 without training collapse.

### 6. Encoder vs Decoder
| Feature      | Encoder                        | Decoder                  |
|--------------|--------------------------------|--------------------------|
| **Sees**     | Full input (bidirectional)     | Past tokens only (masked)|
| **Used for** | Understanding                  | Generation               |
| **Examples** | BERT, RoBERTa                  | GPT, LLaMA               |
| **Tasks**    | Classification, embeddings     | Text generation          |


Encoder-Decoder (T5, BART) → used for translation, summarization where you map input → output sequence.


## Why Transformers Win
| Property                  | RNN/LSTM              | Transformer                    |
|---------------------------|-----------------------|--------------------------------|
| **Long-range dependency** | Poor                  | Excellent (direct attention)   |
| **Parallelization**       | None (sequential)     | Full                           |
| **Scalability**           | Hits wall             | Scales with data + compute     |
| **Training speed**        | Slow                  | Fast                           |