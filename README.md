# LightGBM LambdaRank vs embedding+MLP for conversational entity retrieval

## Motivation
The paper "Benchmark and neural architecture for conversational entity retrieval from a knowledge graph" (2024) introduces a benchmark and a neural architecture for finding the answer entity of each turn in a conversation grounded in a knowledge graph. Its benchmark data could not be confirmed as public, so this experiment uses a simulated stand-in over a public graph. It asks whether a cheap, tuned feature-based reranker can compete with a neural-style scorer.

## Question
Can a tuned LightGBM LambdaRank reranker on simple graph and dialogue-history features beat a neural-style embedding and MLP scorer at retrieving the answer entity for each turn of a simulated conversation over a public knowledge graph?

## Dataset
Public FB15k-237 training triples (downloaded from GitHub at run time). On top of it, 4-turn conversations are simulated: each turn asks for one relation of a focus entity, which is either the previous answer (follow-up) or an earlier entity (topic switch). Candidates are the 1-hop neighbours of all entities mentioned so far. Conversations are split into train and test sets with no overlap. This is **not** the paper's own dataset.

## Baseline vs improved
- **Baseline:** SVD entity (and relation) embeddings of the graph, with an sklearn `MLPClassifier` scoring each (history, relation, candidate) triple. This is a generic neural-style scorer, not the paper's architecture.
- **Improved:** LightGBM LambdaRank on graph and history features (edge relation match, recency of the source entity, ambiguity, degree, already mentioned).

Both use identical candidates, splits, negative sampling, CV-based tuning on train only, and evaluation code. The test set is used once, with seeds 42, 43 and 44. The main metric is MRR, with Hits@1 and Recall@10 as secondary metrics.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`) when the script finishes.

## Results (latest run)

On simulated four-turn conversations over FB15k-237 (3,000 test turns), a LightGBM LambdaRank reranker reached 0.984 MRR versus 0.511 for a standard embedding-plus-MLP neural scorer, though the data is simulated and the features were hand-designed for it.

| Metric | Value |
|---|---|
| baseline_score | 0.511 |
| improved_score | 0.9838 |
| improvement | 0.4728 |
| baseline_score_std | 0.01448 |
| improved_score_std | 0.0003342 |
| baseline_hits1 | 0.354 |
| improved_hits1 | 0.9709 |
| baseline_hits1_std | 0.01446 |
| improved_hits1_std | 0.0006285 |
| baseline_recall10 | 0.8551 |
| improved_recall10 | 0.9993 |
| baseline_recall10_std | 0.01446 |
| improved_recall10_std | 0 |
| mean_candidates_per_query | 51.36 |

Run on 2026-10-23. Full output: [results.json](results.json) · [run.log](run.log)

Notes: n_rows/n_train/n_test count conversation turns (queries); graph has 272115 triples. Simulated stand-in, not the paper's CER-KG benchmark. Entities limited to degree <= 60; each asked (focus, relation) has a unique answer and the answer is never already mentioned, which makes 'already mentioned' an artefact-prone feature. Training/CV use the positive plus 20 sampled negatives per query (CV validation also on sampled candidates); test uses all candidates with pessimistic tie handling. The baseline is not given the focus entity and sees the graph only via embeddings; the tuning grids are small (2 configs each). Chosen params: {"baseline": "{'h': (64, 32), 'alpha': 0.001}", "improved": "{'n': 100, 'lr': 0.1, 'leaves': 15}"}. Data is simulated, so results do not transfer directly to real conver

## Reference

This project is motivated by the following paper:

> M Zamiri, Y Qiang, F Nikolaev, D Zhu (2024). *Benchmark and neural architecture for conversational entity retrieval from a knowledge graph*. Proceedings of the ACM. https://dl.acm.org/doi/abs/10.1145/3589334.3645676

Paper author: Prof. Alexander Kotov, Wayne State University (kotov@wayne.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
