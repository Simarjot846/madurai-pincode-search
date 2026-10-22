import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import load
from filters import Predicate, exact_mask
from ivf import IVFIndex
from strategies import exact_scan
from planner import Planner

vecs, meta, cats, titles = load()
C = {c: i for i, c in enumerate(cats)}
rng = np.random.default_rng(1)
queries = vecs[rng.choice(len(vecs), 100, replace=False)]

idx = IVFIndex(nlist=200).build(vecs)
idx.attach_original(vecs)
pl = Planner(idx, meta)

cases = {
    "all (price 0-1e6)":      [Predicate("price", "between", (0, 10**6))],
    "zone=Mumbai":            [Predicate("zone", "==", 0)],
    "price 500-1500":         [Predicate("price", "between", (500, 1500))],
    "Clothing&Madurai":       [Predicate("category", "==", C["Clothing, Shoes & Jewelry"]), Predicate("zone", "==", 22)],
    "zone=Raipur":            [Predicate("zone", "==", 23)],
    "Pet&Madurai":            [Predicate("category", "==", C["Pet Supplies"]), Predicate("zone", "==", 22)],
    "zone=Leh":               [Predicate("zone", "==", 29)],
}
K = 10
print(f"{'filter':20s} {'true%':>7s} {'est%':>7s} {'src':>8s} {'strategy':22s} {'recall':>6s} {'p50ms':>6s} {'p95ms':>6s}")
for name, preds in cases.items():
    mask = exact_mask(meta, preds)
    strat, est, src = pl.choose(preds)
    rec, lat = [], []
    for q in queries:
        truth = set(exact_scan(idx, q, K, mask)[0].tolist())
        t0 = time.perf_counter()
        ids, _, _ = pl.search(q, K, preds)
        lat.append((time.perf_counter() - t0) * 1000)
        rec.append(len(truth & set(ids.tolist())) / max(len(truth), 1))
    print(f"{name:20s} {mask.mean()*100:7.3f} {est*100:7.3f} {src:>8s} {strat:22s} {np.mean(rec):6.3f} {np.percentile(lat,50):6.2f} {np.percentile(lat,95):6.2f}")