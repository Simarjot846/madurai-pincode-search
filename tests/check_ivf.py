import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ivf import IVFIndex

vecs = np.load("data/processed/vectors.npy")
NQ = 300
queries, base = vecs[:NQ], vecs[NQ:]     # queries are held out of the index

idx = IVFIndex(nlist=200).build(base)
s = idx.sizes()
print("partition sizes  min/median/max:", s.min(), int(np.median(s)), s.max())

K = 10
truth = [set(np.argsort(-(base @ q))[:K]) for q in queries]

for nprobe in [1, 4, 8, 16, 32, 64]:
    rec, lat = [], []
    for q, t in zip(queries, truth):
        t0 = time.perf_counter()
        ids, _ = idx.search(q, K, nprobe)
        lat.append((time.perf_counter() - t0) * 1000)
        rec.append(len(t & set(ids.tolist())) / K)
    print(f"nprobe={nprobe:3d}  recall@10={np.mean(rec):.3f}  p50={np.percentile(lat, 50):.2f} ms")