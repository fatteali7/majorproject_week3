Project Log
-----------

Day 1–2:
Baseline ResNet50 extractor tested successfully.

S2 output:
(4, 2048)

SAR output:
(4, 2048)

Day 3–4:
FAISS IndexFlatL2 tested.

Gallery vectors:
1000

Embedding dimension:
512

Top-k:
10

Query time:
0.716 ms

Result:
FAISS test passed.

## Baseline FAISS

- Dataset: BigEarthNet-S2 validation
- Validation patches: 4500
- Embedding dimension: 2048
- FAISS index: IndexFlatL2
- Gallery size: 4500
- Query: 1 patch
- Top-K: 10
- Query time: 9.679 ms


## FAISS Baseline Retrieval

- Dataset: BigEarthNet-S2 validation
- Validation patches: 4500
- Embedding dimension: 2048
- FAISS index: IndexFlatL2
- Gallery size: 4500
- Query: Validation embedding 0
- Top-K: 10
- Index build time: 15.898 ms
- Query time: 8.791 ms
- Status: PASSED



============================================================
BASELINE RETRIEVAL RESULTS
============================================================
Number of queries: 4500
Gallery size: 4500
Embedding dimension: 2048
Top-K: 10

Baseline mAP@10: 0.966140
Evaluation time: 16.74 seconds
============================================================

## Baseline Retrieval Evaluation

- Validation queries: 4500
- Gallery size: 4500
- Encoder: Pretrained ResNet-50
- Embedding dimension: 2048
- FAISS index: IndexFlatL2
- Top-K: 10
- Index build time: 22.581 ms
- Evaluation time: 24.14 seconds
- Baseline mAP@10: 0.966140
- Baseline mAP@10 (%): 96.614%
- Relevance criterion: Retrieved patch is relevant when it shares at least one BigEarthNet label with the query.