"""Step 7: mutate the catalogue (10% delete, 5% shifted insert) and compare fresh vs stale planners."""
import time
import numpy as np
from filters import exact_mask
from strategies import exact_scan


def mutate(vecs, meta, cats, rng, del_frac=0.10, ins_frac=0.05,
           hot_cat="Pet Supplies", hot_zone=22):
    n = len(vecs)
    C = {c: i for i, c in enumerate(cats)}

    # 10% delete
    keep = np.ones(n, dtype=bool)
    keep[rng.choice(n, int(n * del_frac), replace=False)] = False

    # 5% shifted insert: clones of Pet Supplies items, all placed in one zone, pricier
    n_ins = int(n * ins_frac)
    pool = np.flatnonzero(np.asarray(meta["category"]) == C[hot_cat])
    src = rng.choice(pool, n_ins, replace=True)
    new_vecs = vecs[src] + rng.normal(0, 0.03, size=(n_ins, vecs.shape[1])).astype(np.float32)
    new_vecs /= np.linalg.norm(new_vecs, axis=1, keepdims=True)

    vecs2 = np.vstack([vecs[keep], new_vecs]).astype(vecs.dtype)
    meta2 = {}
    for key, arr in meta.items():
        arr = np.asarray(arr)
        if len(arr) != n:                      # not a per-row column, copy as is
            meta2[key] = arr
            continue
        ins = arr[src].copy()
        if key == "category":
            ins[:] = C[hot_cat]
        elif key == "zone":
            ins[:] = hot_zone
        elif key == "price":
            ins = (ins * 1.5).astype(arr.dtype)
        meta2[key] = np.concatenate([arr[keep], ins])
    return vecs2, meta2


def check_cases(planners, idx, meta, cases, queries, k=10):
    """planners: {name: Planner}. Prints est, strategy, recall, p50 for each filter."""
    print(f"{'filter':18s} {'true%':>7s} {'planner':8s} {'est%':>7s} {'src':>8s} "
          f"{'strategy':22s} {'recall':>6s} {'p50ms':>7s}")
    for name, preds in cases.items():
        mask = exact_mask(meta, preds)
        truths = [set(exact_scan(idx, q, k, mask)[0].tolist()) for q in queries]
        for pname, pl in planners.items():
            strat, est, src = pl.choose(preds)
            rec, lat = [], []
            pl.search(queries[0], k, preds)                    # warm-up
            for q, truth in zip(queries, truths):
                t = time.perf_counter()
                ids, _, _ = pl.search(q, k, preds)
                lat.append((time.perf_counter() - t) * 1000)
                rec.append(len(truth & set(ids.tolist())) / max(len(truth), 1))
            print(f"{name:18s} {mask.mean()*100:7.3f} {pname:8s} {est*100:7.3f} {src:>8s} "
                  f"{strat:22s} {np.mean(rec):6.3f} {np.median(lat):7.2f}")
        print()