import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import load
from ivf import IVFIndex
from workload import build_workload
from planner import Planner
from break_planner import BrokenPlanner, compare_modes, worst_correlated_pairs

vecs, meta, cats, titles = load()
idx = IVFIndex(nlist=200).build(vecs)
idx.attach_original(vecs)

good = Planner(idx, meta)
planners = {"good": good}
for m in ("indep_only", "always_post", "no_exact", "inverted"):
    planners[m] = BrokenPlanner(idx, meta, m)

work = build_workload(vecs, meta, len(cats), 30)
worst_correlated_pairs(good, meta, len(cats), 30)
print()
compare_modes(idx, meta, work, planners)