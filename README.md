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

## Reference

This project is motivated by the following paper:

> M Zamiri, Y Qiang, F Nikolaev, D Zhu (2024). *Benchmark and neural architecture for conversational entity retrieval from a knowledge graph*. Proceedings of the ACM. https://dl.acm.org/doi/abs/10.1145/3589334.3645676

Paper author: Prof. Alexander Kotov, Wayne State University (kotov@wayne.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
