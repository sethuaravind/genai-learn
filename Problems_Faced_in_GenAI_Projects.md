# Problems Faced in GenAI Projects


## Returning Signed URLs

### Problem
- Signed URLs (Azure Blob, GCP Storage, S3, etc.) contain long query strings and characters that LLMs can alter when the URL is included verbatim in prompts or generated text, which breaks the link.

### Solution
- Replace each signed URL in the prompt with a stable placeholder before sending to the LLM (for example `[[SIGNED_URL_1]]`).
- Maintain a mapping (placeholder → original URL) in a secure in-memory store or cache accessible to the post-processing step.
- After receiving the model response, locate placeholders (e.g. with a simple regex) and substitute them back with the original signed URLs from the mapping.

Notes and best practices:
- Use a non-natural placeholder format (brackets or a unique prefix) to reduce accidental changes by the model.
- Generate deterministic placeholder IDs (incrementing or hashed) to make replacement straightforward.
- Validate and sanitize replaced URLs before returning them to callers.
- For enhanced security, keep signed URLs short-lived and store the real URLs in a secure store rather than logging them.