import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import load
from filters import Predicate, exact_mask
from planner import Planner

vecs, meta, cats, titles = load()
pl = Planner(None, meta)          # idx not needed, we only call estimate()
n = len(meta["price"])

def decide(rows):                 # at this scale there are only two tiers
    return "exact" if rows <= pl.t_exact_rows else "post"

total = flips_indep = flips_planner = 0
for c in range(len(cats)):
    for z in range(30):
        preds = [Predicate("category", "==", c), Predicate("zone", "==", z)]
        truth = decide(int(exact_mask(meta, preds).sum()))
        flips_indep += decide(pl.indep.estimate(preds) * n) != truth
        flips_planner += decide(pl.estimate(preds)[0] * n) != truth
        total += 1
print(f"pairs: {total}")
print(f"decision wrong with independence estimate: {flips_indep}")
print(f"decision wrong with planner estimate:      {flips_planner}")