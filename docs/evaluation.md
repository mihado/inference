# Evaluation

The numbers below come from the codex retrieval evaluation.

- The data set has 7,277 nodes from New York, DC, and Maryland.
- The question set has 500 held-out questions.
- The retrieval is hybrid with alpha 0.5 and 30 candidates.
- The index is `voyage-4-nano` at 1024 dimensions.
- The metrics are recall and MRR. A higher number is better.

## Embedding model, no rerank

| Model | Dimensions | recall@1 | recall@5 | MRR | Index speed |
| --- | --- | --- | --- | --- | --- |
| `voyage-4-nano` | 1024 | 0.564 | 0.906 | 0.710 | ~214 nodes/s |
| `voyage-4-nano` | 2048 | 0.560 | 0.902 | 0.709 | ~214 nodes/s |
| `Qwen/Qwen3-Embedding-0.6B` | 1024 | 0.562 | 0.880 | 0.696 | ~123 nodes/s |
| `Qwen/Qwen3-Embedding-4B` | 2560 | 0.566 | 0.882 | 0.702 | ~18 nodes/s |

The 1024 and 2048 results are equal. So the 1024 index is the default. It is half the size.

## Reranker

A reranker reads the 30 candidates and puts them in a new order.

| Reranker | recall@1 | recall@5 | MRR |
| --- | --- | --- | --- |
| none | 0.564 | 0.904 | 0.710 |
| Voyage `rerank-3-lite` | 0.796 | 0.962 | 0.873 |
| TypeSafe `jev-latest` | 0.752 | 0.960 | 0.848 |
| `Alibaba-NLP/gte-reranker-modernbert-base` | 0.724 | 0.934 | 0.822 |
| `BAAI/bge-reranker-v2-m3` | 0.676 | 0.930 | 0.788 |
| `ibm-granite/granite-embedding-reranker-english-r2` | 0.660 | 0.928 | 0.780 |
| `Alibaba-NLP/gte-multilingual-reranker-base` | 0.654 | 0.916 | 0.768 |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | 0.684 | 0.906 | 0.780 |

Results:

- You must use a reranker. The best local reranker adds 16 points to recall@1 and 3 points to recall@5.
- The best local reranker is `Alibaba-NLP/gte-reranker-modernbert-base`. The `reranker` server runs this model.
- Voyage `rerank-3-lite` is the best reranker. It is the MVP choice.
- A difference of 0.002 is noise. One question of 500 causes it.
