import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import load
from filters import Predicate, exact_mask
from ivf import IVFIndex
from strategies import exact_scan, postfilter_adaptive, partition_restricted

vecs, meta, cats, titles = load()
C = {c: i for i, c in enumerate(cats)}
rng = np.random.default_rng(1)
NQ = 100
qi = rng.choice(len(vecs), NQ, replace=False)
keep = np.ones(len(vecs), bool); keep[qi] = False
queries = vecs[qi]

idx = IVFIndex(nlist=200).build(vecs)
idx.attach_original(vecs)

cases = {
    "zone=Mumbai (18%)":      [Predicate("zone", "==", 0)],
    "Clothing&Madurai (2%)":  [Predicate("category", "==", C["Clothing, Shoes & Jewelry"]), Predicate("zone", "==", 22)],
    "Pet&Madurai (0.02%)":    [Predicate("category", "==", C["Pet Supplies"]), Predicate("zone", "==", 22)],
}
K = 10
for name, preds in cases.items():
    mask = exact_mask(meta, preds) & keep
    print(f"\n{name}: {mask.sum()} matches")
    for sname, fn in [("exact", exact_scan), ("postfilter", postfilter_adaptive), ("part_restr", partition_restricted)]:
        rec, lat = [], []
        for q in queries:
            truth = set(exact_scan(idx, q, K, mask)[0].tolist())
            t0 = time.perf_counter()
            ids, _ = fn(idx, q, K, mask)
            lat.append((time.perf_counter() - t0) * 1000)
            rec.append(len(truth & set(ids.tolist())) / max(len(truth), 1))
        print(f"  {sname:11s} recall={np.mean(rec):.3f}  p50={np.percentile(lat,50):.2f} ms  p95={np.percentile(lat,95):.2f} ms")