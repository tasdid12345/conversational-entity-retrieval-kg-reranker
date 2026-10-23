"""LightGBM LambdaRank reranker vs SVD-embedding + MLP scorer on simulated conversational entity retrieval (FB15k-237)."""
import io, json, os, sys, time
from collections import Counter
import numpy as np, pandas as pd, requests, lightgbm as lgb
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42; SEEDS = [42, 43, 44]
URL = "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237/train.txt"
N_CONV, TURNS, MAXDEG, NNEG, DIM, RDIM = 3000, 4, 60, 20, 32, 16
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

# ---------- data: download, graph, embeddings ----------
def download():
    for a in range(5):
        r = requests.get(URL, timeout=60)
        if r.status_code in (429, 500, 502, 503, 504): time.sleep(2 ** a); continue
        r.raise_for_status(); return r.text
    raise RuntimeError("download failed")

df = pd.read_csv(io.StringIO(download()), sep="\t", header=None, names=["h", "r", "t"])
codes, ents = pd.factorize(pd.concat([df.h, df.t])); H, T = codes[:len(df)], codes[len(df):]
RL, rels = pd.factorize(df.r); E_N, R = len(ents), len(rels)
adj = [[] for _ in range(E_N)]  # directed edges: relation r forward, r+R inverse
for h, r, t in zip(H, RL, T): adj[h].append((r, t)); adj[t].append((r + R, h))
relcnt = [Counter(r for r, _ in a) for a in adj]; deg = np.array([len(a) for a in adj])
log(f"graph: {len(df)} triples, {E_N} entities, {R} relations")
A = sparse.coo_matrix((np.ones(2 * len(H)), (np.r_[H, T], np.r_[T, H])), shape=(E_N, E_N)).tocsr()
dn = sparse.diags(1 / np.sqrt(np.maximum(deg, 1)))
EMB = TruncatedSVD(DIM, random_state=SEED).fit_transform(dn @ A @ dn)  # entity embeddings
rows = np.r_[RL, RL + R, RL, RL + R]; cols = np.r_[H, T, E_N + T, E_N + H]
M = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(2 * R, 2 * E_N)).tocsr(); M.data = np.log1p(M.data)
REMB = TruncatedSVD(RDIM, random_state=SEED).fit_transform(M)  # directed-relation embeddings

# ---------- conversation simulation ----------
def make_query(cid, hist, q, ans, follow):
    """Candidates = 1-hop neighbours of all mentioned entities; graph/history features per candidate."""
    info, L = {}, len(hist)
    for i, m in enumerate(hist):
        rec = L - 1 - i
        for r, c in adj[m]:
            d = info.setdefault(c, [0, 99, 99, 0]); d[3] += 1; d[2] = min(d[2], rec)
            if r == q: d[0] += 1; d[1] = min(d[1], rec)
    cands = np.array(list(info)); a = np.array(list(info.values()), float)
    inv = (q + R) % (2 * R); m_ = a[:, 0] > 0
    rel_rec = a[:, 1] - (a[m_, 1].min() if m_.any() else 0)
    F = np.column_stack([a, np.full(len(a), m_.sum()), np.log1p(deg[cands]), np.isin(cands, hist),
                         [relcnt[c].get(inv, 0) for c in cands], rel_rec])
    return dict(cid=cid, hist=list(hist), q=q, cands=cands, F=F, pos=int(np.flatnonzero(cands == ans)[0]), follow=follow)

def simulate(rng):
    qs, cid, ok = [], 0, np.flatnonzero((deg >= 3) & (deg <= MAXDEG))
    while cid < N_CONV:
        hist, turns = [int(rng.choice(ok))], []
        for _ in range(TURNS):
            follow = len(hist) == 1 or rng.random() < 0.6  # follow-up on last answer vs topic switch
            f = hist[-1] if follow else hist[int(rng.integers(len(hist) - 1))]
            opts = [(r, t) for r, t in adj[f] if relcnt[f][r] == 1 and t not in hist and 2 <= deg[t] <= MAXDEG]
            if not opts: break
            r, t = opts[int(rng.integers(len(opts)))]
            turns.append((list(hist), r, t, follow)); hist.append(t)
        if len(turns) == TURNS:
            qs += [make_query(cid, *x) for x in turns]; cid += 1
    return qs

rng = np.random.default_rng(SEED)
qs = simulate(rng)
tr_c, te_c = train_test_split(np.arange(N_CONV), test_size=0.25, random_state=SEED)  # split by conversation
te_set = set(te_c.tolist())
train = [q for q in qs if q["cid"] not in te_set]; test = [q for q in qs if q["cid"] in te_set]
for q in train:  # shared negative subsample for training/CV: positive + up to NNEG random negatives
    neg = np.delete(np.arange(len(q["cands"])), q["pos"])
    q["sub"] = np.sort(np.r_[q["pos"], rng.choice(neg, min(NNEG, len(neg)), replace=False)])
log(f"queries: train={len(train)} test={len(test)}, mean cands={np.mean([len(q['cands']) for q in qs]):.0f}")

# ---------- shared evaluation ----------
def evaluate(predict, feat, qs_, full):
    ranks = []
    for i in range(0, len(qs_), 200):
        ch = qs_[i:i + 200]; idx = [np.arange(len(q["cands"])) if full else q["sub"] for q in ch]
        s = predict(np.vstack([feat(q, ix) for q, ix in zip(ch, idx)])); o = 0
        for q, ix in zip(ch, idx):
            sc = s[o:o + len(ix)]; o += len(ix); sp = sc[np.flatnonzero(ix == q["pos"])[0]]
            ranks.append(1 + (sc > sp).sum() + (sc == sp).sum() - 1)  # pessimistic on ties
    r = np.array(ranks)
    return {"mrr": float((1 / r).mean()), "hits1": float((r == 1).mean()), "recall10": float((r <= 10).mean())}

def build(qs_, feat):
    X = np.vstack([feat(q, q["sub"]) for q in qs_])
    y = np.concatenate([q["sub"] == q["pos"] for q in qs_]).astype(int)
    return X, y, [len(q["sub"]) for q in qs_]

# ---------- models ----------
def feat_lgb(q, ix): return q["F"][ix]

def feat_mlp(q, ix):
    h = EMB[q["hist"]]; w = 0.5 ** np.arange(len(h))[::-1]; hm = (w[:, None] * h).sum(0) / w.sum()
    C = EMB[q["cands"][ix]]; ctx = np.concatenate([hm, h[-1], REMB[q["q"]]])
    return np.hstack([np.tile(ctx, (len(C), 1)), C, C * hm, C * h[-1]]).astype(np.float32)

def fit_lgb(qs_, p, seed):
    X, y, g = build(qs_, feat_lgb)
    m = lgb.LGBMRanker(objective="lambdarank", n_estimators=p["n"], learning_rate=p["lr"], num_leaves=p["leaves"],
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    return m.fit(X, y, group=g).predict

def fit_mlp(qs_, p, seed):
    X, y, _ = build(qs_, feat_mlp)
    pipe = make_pipeline(StandardScaler(), MLPClassifier(p["h"], alpha=p["alpha"], batch_size=512, max_iter=30,
                         early_stopping=True, n_iter_no_change=4, random_state=seed)).fit(X, y)
    sc, clf = pipe[0], pipe[1]
    def predict(Xn):  # raw logit (avoids probability saturation ties)
        a = sc.transform(Xn)
        for W, b in zip(clf.coefs_[:-1], clf.intercepts_[:-1]): a = np.maximum(a @ W + b, 0)
        return (a @ clf.coefs_[-1] + clf.intercepts_[-1]).ravel()
    return predict

MODELS = {
    "baseline": (fit_mlp, feat_mlp, [dict(h=(128, 64), alpha=1e-4), dict(h=(64, 32), alpha=1e-3)]),
    "improved": (fit_lgb, feat_lgb, [dict(n=100, lr=0.1, leaves=15), dict(n=300, lr=0.05, leaves=31)]),
}

def cv_mrr(fit, feat, p):  # 3-fold CV on train only, grouped by conversation (validation on sampled candidates)
    sc = []
    for a, b in GroupKFold(3).split(train, groups=[q["cid"] for q in train]):
        pred = fit([train[i] for i in a], p, SEED)
        sc.append(evaluate(pred, feat, [train[i] for i in b], False)["mrr"])
    return float(np.mean(sc))

res, best = {}, {}
for name, (fit, feat, grid) in MODELS.items():
    scores = [cv_mrr(fit, feat, p) for p in grid]; best[name] = grid[int(np.argmax(scores))]
    log(f"{name}: CV MRR per config {np.round(scores, 4).tolist()} -> {best[name]}")
for name, (fit, feat, _) in MODELS.items():  # test set used once, final evaluation
    runs = [evaluate(fit(train, best[name], s), feat, test, True) for s in SEEDS]
    res[name] = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in runs[0]}
    log(f"{name} test: " + ", ".join(f"{k}={v[0]:.4f}" for k, v in res[name].items()))

# ---------- outputs ----------
b, i = res["baseline"], res["improved"]
metrics = {"baseline_score": b["mrr"][0], "improved_score": i["mrr"][0], "improvement": i["mrr"][0] - b["mrr"][0],
           "baseline_score_std": b["mrr"][1], "improved_score_std": i["mrr"][1]}
for k in ("hits1", "recall10"):
    metrics.update({f"baseline_{k}": b[k][0], f"improved_{k}": i[k][0], f"baseline_{k}_std": b[k][1], f"improved_{k}_std": i[k][1]})
metrics["mean_candidates_per_query"] = float(np.mean([len(q["cands"]) for q in qs]))
if not all(np.isfinite(v) for v in metrics.values()) or not test: sys.exit("no meaningful metrics")
os.makedirs("figures", exist_ok=True)
ks = ["mrr", "hits1", "recall10"]; x = np.arange(3)
plt.bar(x - 0.2, [b[k][0] for k in ks], 0.4, yerr=[b[k][1] for k in ks], label="SVD+MLP baseline")
plt.bar(x + 0.2, [i[k][0] for k in ks], 0.4, yerr=[i[k][1] for k in ks], label="LightGBM LambdaRank")
plt.xticks(x, ["MRR", "Hits@1", "Recall@10"]); plt.legend(); plt.title("Test set, mean of 3 seeds")
plt.savefig("figures/comparison.png", dpi=120, bbox_inches="tight")
out = {
    "question": "Can a tuned LightGBM LambdaRank reranker on simple graph and dialogue-history features beat an embedding+MLP scorer at retrieving the answer entity per turn of a simulated conversation over a public KG?",
    "method": f"Simulated {N_CONV} four-turn conversations over FB15k-237 (follow-up vs topic-switch focus); candidates are 1-hop neighbours of all mentioned entities; conversation-level 75/25 split; hyper-parameters chosen by 3-fold grouped CV on train; both models evaluated once on test over all candidates, seeds 42-44.",
    "dataset": {"name": "FB15k-237 train triples, simulated conversational entity retrieval", "source_url": URL,
                "n_rows": len(qs), "n_train": len(train), "n_test": len(test), "same_as_paper": False,
                "baseline_is_papers_method": False},
    "metric_name": "MRR", "higher_is_better": True,
    "baseline": "Truncated-SVD entity/relation embeddings of the graph feeding an sklearn MLPClassifier that scores (history, relation, candidate) features.",
    "improved": "LightGBM LambdaRank reranker on graph/history features: relation match, recency of the matching source entity, ambiguity, degree, already-mentioned, relation-conditioned in-degree.",
    "metrics": metrics,
    "notes": (f"n_rows/n_train/n_test count conversation turns (queries); graph has {len(df)} triples. Simulated stand-in, not the paper's CER-KG benchmark. "
              f"Entities limited to degree <= {MAXDEG}; each asked (focus, relation) has a unique answer and the answer is never already mentioned, "
              "which makes 'already mentioned' an artefact-prone feature. Training/CV use the positive plus "
              f"{NNEG} sampled negatives per query (CV validation also on sampled candidates); test uses all candidates with pessimistic tie handling. "
              "The baseline is not given the focus entity and sees the graph only via embeddings; the tuning grids are small (2 configs each). "
              f"Chosen params: {json.dumps({k: str(v) for k, v in best.items()})}. Data is simulated, so results do not transfer directly to real conversations."),
}
with open("results.json", "w", encoding="utf-8") as f: json.dump(out, f, indent=2)
log("done; wrote results.json")
