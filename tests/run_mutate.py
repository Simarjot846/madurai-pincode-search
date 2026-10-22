import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import load
from filters import Predicate, SelectivityEstimator, SampleEstimator
from ivf import IVFIndex
from workload import build_workload
from planner import Planner
from benchmark import run_all
from mutate import mutate, check_cases

vecs, meta, cats, titles = load()
C = {c: i for i, c in enumerate(cats)}
rng = np.random.default_rng(7)

vecs2, meta2 = mutate(vecs, meta, cats, rng)
print(f"before: {len(vecs)} items, after: {len(vecs2)} items\n")

idx2 = IVFIndex(nlist=200).build(vecs2)
idx2.attach_original(vecs2)

fresh = Planner(idx2, meta2)                       # estimators built on mutated data
stale = Planner(idx2, meta2)
stale.indep = SelectivityEstimator(meta)           # estimators still from the OLD data
stale.sample = SampleEstimator(meta)

qs = vecs2[rng.choice(len(vecs2), 40, replace=False)]
cases = {
    "Pet&Madurai (hot)": [Predicate("category", "==", C["Pet Supplies"]), Predicate("zone", "==", 22)],
    "Clothing&Madurai":  [Predicate("category", "==", C["Clothing, Shoes & Jewelry"]), Predicate("zone", "==", 22)],
    "zone=Mumbai":       [Predicate("zone", "==", 0)],
    "zone=Leh":          [Predicate("zone", "==", 29)],
}
check_cases({"fresh": fresh, "stale": stale}, idx2, meta2, cases, qs)

print("=== full benchmark on mutated data (fresh planner) ===")
work = build_workload(vecs2, meta2, len(cats), 30)
run_all(fresh, meta2, work)