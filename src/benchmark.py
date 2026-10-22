"""Benchmark: planner vs pure post-filter vs pure pre-filter, per selectivity bucket."""
import time
import numpy as np
from filters import exact_mask
from baselines import pure_postfilter, pure_prefilter
from workload import BUCKETS


def _timed(fn, reps):
    """Run fn `reps` times, return (last output, median ms)."""
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t) * 1000)
    return out, float(np.median(ts))


def run_bucket(planner, meta, items, k=10, reps=3):
    idx = planner.idx
    methods = {
        "planner":    lambda q, p: planner.search(q, k, p)[:2],
        "postfilter": lambda q, p: pure_postfilter(idx, q, k, exact_mask(meta, p)),
        "prefilter":  lambda q, p: pure_prefilter(idx, q, k, exact_mask(meta, p)),
    }
    res = {m: {"rec": [], "lat": []} for m in methods}
    strats = {}

    # warm-up: a few untimed runs so caches/BLAS are hot
    for preds, qs in items[:2]:
        for q in qs[:3]:
            for f in methods.values():
                f(q, preds)

    for preds, qs in items:
        s = planner.choose(preds)[0]
        strats[s] = strats.get(s, 0) + 1
        for q in qs:
            truth, _ = pure_prefilter(idx, q, k, exact_mask(meta, preds))
            if len(truth) == 0:
                continue
            tset = set(truth.tolist())
            for name, f in methods.items():
                (ids, _), ms = _timed(lambda: f(q, preds), reps)
                res[name]["rec"].append(len(tset & set(ids.tolist())) / len(tset))
                res[name]["lat"].append(ms)
    return res, strats


def run_all(planner, meta, work, k=10, reps=3):
    print(f"{'bucket':8s} {'method':11s} {'recall':>7s} {'p50ms':>8s} {'p95ms':>8s}   planner strategies")
    for b in BUCKETS:
        res, strats = run_bucket(planner, meta, work[b], k, reps)
        for i, (name, r) in enumerate(res.items()):
            lat = np.array(r["lat"])
            extra = str(strats) if i == 0 else ""
            print(f"{b:8s} {name:11s} {np.mean(r['rec']):7.3f} "
                  f"{np.percentile(lat, 50):8.2f} {np.percentile(lat, 95):8.2f}   {extra}")
        print()