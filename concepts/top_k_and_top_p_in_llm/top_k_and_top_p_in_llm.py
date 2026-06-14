from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path().resolve() / ".env")

client = OpenAI()  # reads OPENAI_API_KEY from env

prompt = "The future of artificial intelligence is"

# --- Low top_p: focused, deterministic, RAG-friendly ---
response_low = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    top_p=0.1,   # only considers tokens in top 10% cumulative probability mass
    max_tokens=50,
)

# --- High top_p: more diverse, creative ---
response_high = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    top_p=0.95,  # considers a much wider set of tokens
    max_tokens=50,
)

print("Low top_p (0.1):")
print(response_low.choices[0].message.content)
print("\nHigh top_p (0.95):")
print(response_high.choices[0].message.content)