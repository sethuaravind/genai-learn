# Top-k and Top-p (Nucleus) sampling

Top-k and Top-p are sampling parameters used by Large Language Models (LLMs) to control randomness and creativity when generating text. They determine which subset of the model's predicted tokens are eligible for selection at each step.

## Top-k sampling

- How it works: The model computes probabilities for all possible next tokens, sorts them, and keeps only the top `k` tokens. All other tokens are discarded and cannot be chosen.
- Effect: When `k = 1` the model becomes deterministic (greedy decoding). Smaller `k` (e.g., `k = 10`) produces safer, more focused text; larger `k` (e.g., `k = 50`) increases variety and creativity.

## Top-p sampling (nucleus sampling)

- How it works: Instead of a fixed token count, top-p selects the smallest set of top tokens whose cumulative probability is at least `p` (for example, `p = 0.9` keeps the smallest group that sums to 90% of the probability mass).
- Effect: Top-p adapts to the model's confidence. If the model is confident, the candidate pool will be small; if uncertain, the pool grows. This gives a balance between coherence and diversity without including many very low-probability tokens.

## Quick comparison

| Feature | Top-k | Top-p (Nucleus) |
|---|---:|---|
| Selection | Fixed number of tokens (`k`) | Dynamic set by cumulative probability (`p`) |
| Adaptability | Rigid: always `k` tokens | Fluid: fewer tokens when confident, more when uncertain |
| Best for | Structured tasks, deterministic outputs | Open-ended text, creative writing |

## Usage and recommendations

- Temperature, `top_k`, and `top_p` are often used together. `temperature` controls how flat or sharp the distribution is before sampling; `top_k`/`top_p` control which tokens are eligible.
- Common practical settings:
	- `temperature = 0.7`, `top_p = 0.9` — good default for creative-but-coherent text.
	- `temperature = 0.0`, `top_k = 1` — deterministic, for factual or programmatic outputs.

### Example (OpenAI client)

```python
from langchain_openai import OpenAI

llm = OpenAI(
		model="gpt-5.4-mini-2026-03-17",
		temperature=0.7,
		top_p=0.9,
		max_tokens=512,
)
```

### Note on retrieval vs. sampling

The term "top-k" is also used in retrieval (e.g., returning the top `k` documents). That is a different use of `k`: retrieval `k` limits the number of retrieved documents, while sampling `k` limits token candidates during generation.

---

For further reading, consult original papers on nucleus sampling and modern tokenizer/sampling techniques.