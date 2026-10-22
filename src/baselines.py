"""The two baselines the search team already tried."""
import numpy as np


def pure_postfilter(idx, q, k, mask, nprobe=16, fetch=100):
    """Search team attempt 1: take the 100 nearest via the index, then drop non-matching ones."""
    ids, sims = idx.search(q, fetch, nprobe)
    keep = mask[ids]
    ids, sims = ids[keep][:k], sims[keep][:k]
    return ids, sims


def pure_prefilter(idx, q, k, mask):
    """Search team attempt 2: filter first, then exact scan of the survivors."""
    rows = np.flatnonzero(mask)
    if len(rows) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    sims = idx.orig_vecs[rows] @ q
    kk = min(k, len(rows))
    top = np.argpartition(-sims, kk - 1)[:kk]
    top = top[np.argsort(-sims[top])]
    return rows[top], sims[top]