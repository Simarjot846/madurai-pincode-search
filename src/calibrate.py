"""Sweep every strategy variant on every bucket, so thresholds come from measurements."""
import time
from functools import partial
import numpy as np
from filters import exact_mask
from strategies import exact_scan, postfilter_adaptive, partition_restricted
from workload import BUCKETS

VARIANTS = {"exact": exact_scan}
for s in (32, 40, 48, 64):
    VARIANTS[f"post_s{s}"] = partial(postfilter_adaptive, start=s)
for n in (32, 64, 96, 128):
    VARIANTS[f"part_n{n}"] = partial(partition_restricted, nprobe=n)


def run_filter(idx, meta, preds, qs, k, reps):
    """Returns {variant: (mean recall, [per-query median ms])}. Mask build is inside the timing."""
    mask = exact_mask(meta, preds)
    for fn in VARIANTS.values():                      # warm-up, untimed
        fn(idx, qs[0], k, mask)
    out = {v: ([], []) for v in VARIANTS}
    for q in qs:
        truth = set(exact_scan(idx, q, k, mask)[0].tolist())
        if not truth:
            continue
        for v, fn in VARIANTS.items():
            ts = []
            for _ in range(reps):
                t = time.perf_counter()
                ids, _ = fn(idx, q, k, exact_mask(meta, preds))
                ts.append((time.perf_counter() - t) * 1000)
            out[v][0].append(len(truth & set(ids.tolist())) / len(truth))
            out[v][1].append(float(np.median(ts)))
    return out


def calibrate(idx, meta, work, k=10, reps=3, n_filters=4, n_q=8, min_recall=0.95):
    n = len(meta["price"])
    per_filter = []
    print(f"{'bucket':8s} {'variant':10s} {'recall':>7s} {'p50ms':>8s}")
    for b in BUCKETS:
        agg = {v: ([], []) for v in VARIANTS}
        for preds, qs in work[b][:n_filters]:
            res = run_filter(idx, meta, preds, qs[:n_q], k, reps)
            m = int(exact_mask(meta, preds).sum())
            per_filter.append((b, m, res))
            for v, (r, l) in res.items():
                agg[v][0].extend(r); agg[v][1].extend(l)
        for v, (r, l) in agg.items():
            print(f"{b:8s} {v:10s} {np.mean(r):7.3f} {np.percentile(l, 50):8.2f}")
        print()

    print(f"Per filter: fastest variant with recall >= {min_recall}")
    print(f"{'bucket':8s} {'matches':>8s} {'sel%':>7s} {'best':10s} {'ms':>7s} {'exact_ms':>9s}")
    for b, m, res in sorted(per_filter, key=lambda x: x[1]):
        ok = {v: np.median(l) for v, (r, l) in res.items() if np.mean(r) >= min_recall}
        best = min(ok, key=ok.get) if ok else "none"
        print(f"{b:8s} {m:8d} {m / n * 100:7.2f} {best:10s} {ok.get(best, float('nan')):7.2f} "
              f"{np.median(res['exact'][1]):9.2f}")