# Jina

- Jina v3.5 is a listwise reranker: Qwen3-0.6B plus custom modeling code that ranks many documents jointly in one forward pass, with relevance-ordered results out of the box.
- Own Python service in `jina/`, in the `jina` profile, on GPU 0 — TEI cannot host custom modeling code.
- Non-commercial weights (CC-BY-NC-4.0): local dev and eval only, never serving. Bake-off reference, not a serving candidate.
- `POST /rerank` takes TEI's `{query, texts}` (64 max) straight into the native call; no router change was needed.

The sibling `jina-embed` service serves `jina-embeddings-v3` (XLM-R embedder, 1024 dims) behind OpenAI's `/v1/embeddings`, which the router already forwards verbatim — also no router change. Same non-commercial terms, same local-dev scope. Its transformers pin stays on the 4.x line: the custom modeling code targets the 4.x module API and breaks on 5.x.
