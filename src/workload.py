"""Builds the benchmark workload: filters grouped into 5 selectivity buckets, with realistic queries."""
import numpy as np
from filters import Predicate, exact_mask

BUCKETS = ["~100%", "~10%", "~1%", "~0.1%", "handful"]
RANGES = {  
    "~100%": (0.90, 1.00), "~10%": (0.05, 0.20), "~1%": (0.005, 0.02),
    "~0.1%": (0.0005, 0.002),
}


def _bucket_of(n_match, n):
    if 5 <= n_match <= 40:
        return "handful"
    f = n_match / n
    for b, (lo, hi) in RANGES.items():
        if lo <= f <= hi:
            return b
    return None


def candidate_filters(meta, n_cats, n_zones, rng, n_try=4000):
    n = len(meta["price"])
    ps = np.sort(meta["price"])      
    out = []
    pmax = int(meta["price"].max())
    for _ in range(n_try):
        kind = rng.integers(0, 6)      
        preds = []
        if kind == 5:
            w = 10 ** rng.uniform(-3.3, 0)
            start = rng.uniform(0, 1 - w)
            lo = int(ps[int(start * n)])
            hi = int(ps[min(n - 1, int((start + w) * n))])
            preds = [Predicate("price", "between", (lo, hi))]
        elif kind == 0:
            preds = [Predicate("price", "between", (0, pmax))]
        elif kind == 1:
            preds = [Predicate("zone", "==", int(rng.integers(0, n_zones)))]
        elif kind == 2:
            preds = [Predicate("category", "==", int(rng.integers(0, n_cats))),
                     Predicate("zone", "==", int(rng.integers(0, n_zones)))]
        elif kind == 3:
            lo = int(rng.integers(49, 3000)); hi = lo + int(rng.integers(100, 8000))
            preds = [Predicate("category", "==", int(rng.integers(0, n_cats))),
                     Predicate("price", "between", (lo, hi))]
        else:
            lo = int(rng.integers(49, 2000)); hi = lo + int(rng.integers(100, 6000))
            preds = [Predicate("category", "==", int(rng.integers(0, n_cats))),
                     Predicate("zone", "==", int(rng.integers(0, n_zones))),
                     Predicate("price", "between", (lo, hi))]
        out.append(preds)
    return out


def build_workload(vecs, meta, n_cats, n_zones, per_bucket=8, nq=15, seed=11):
    """Returns {bucket: [(preds, query_matrix), ...]}. Queries are real items that pass the filter,
    lightly perturbed, so the query sits near its matches (like a user searching inside a category)."""
    rng = np.random.default_rng(seed)
    n = len(meta["price"])
    work = {b: [] for b in BUCKETS}
    seen = set()
    for preds in candidate_filters(meta, n_cats, n_zones, rng):
        mask = exact_mask(meta, preds)
        m = int(mask.sum())
        b = _bucket_of(m, n)
        key = (b, tuple(map(repr, preds)))
        if b is None or key in seen or len(work[b]) >= per_bucket:
            continue
        seen.add(key)
        rows = np.flatnonzero(mask)
        pick = rng.choice(rows, size=min(nq, len(rows)), replace=False)
        qs = vecs[pick] + rng.normal(0, 0.02, size=(len(pick), vecs.shape[1])).astype(np.float32)
        qs /= np.linalg.norm(qs, axis=1, keepdims=True)
        work[b].append((preds, qs))
    return work