# Jina

- Jina v3.5 is a listwise reranker: Qwen3-0.6B plus custom modeling code that ranks many documents jointly in one forward pass, with relevance-ordered results out of the box.
- Own Python service in `jina/`, in the `jina` profile, on GPU 0 — TEI cannot host custom modeling code.
- Non-commercial weights (CC-BY-NC-4.0): local dev and eval only, never serving. Bake-off reference, not a serving candidate.
- `POST /rerank` takes TEI's `{query, texts}` (64 max) straight into the native call; no router change was needed.
- Both profiles take a second replica: `make up-jina CONCURRENCY=2` starts `jina-b` (8094) and `make up-jina-embed CONCURRENCY=2` starts `jina-embed-b` (8092), each on the other card; the router alternates requests between the two.

The sibling `jina-embed` service serves `jina-embeddings-v5-omni-small` — a Qwen3-family multimodal embedder (~1.74B, 1024 dims, Matryoshka; text, image, and audio/video under the omni modality) — behind OpenAI's `/v1/embeddings`, which the router already forwards verbatim, also no router change. It loads the retrieval task and the vision towers, so callers pick `input_type` query/document instead of hand-building the model's prefixes, pass images as base64 `data:` URL items, and can truncate with `dimensions`. Same non-commercial terms, same local-dev scope.
