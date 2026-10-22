"""The three filtered-search strategies. Each returns (ids, sims) over original row ids."""
import numpy as np


def exact_scan(idx, q, k, mask):
    rows = np.flatnonzero(mask)
    if len(rows) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    sims = idx.orig_vecs[rows] @ q
    kk = min(k, len(rows))
    top = np.argpartition(-sims, kk - 1)[:kk]
    top = top[np.argsort(-sims[top])]
    return rows[top], sims[top]


def postfilter_adaptive(idx, q, k, mask, start=64, max_nprobe=None):
    """Probe at least `start` partitions, then keep doubling until k matches are found."""
    max_nprobe = max_nprobe or idx.nlist
    order = idx.probe(q, idx.nlist)              # all partitions, nearest first
    nprobe = start
    while True:
        ids, sims = idx.search_parts(q, k, order[:nprobe], mask)
        if len(ids) >= k or nprobe >= max_nprobe:
            return ids, sims
        nprobe = min(nprobe * 2, max_nprobe)


def partition_restricted(idx, q, k, mask, nprobe=64):
    """Only partitions holding at least one matching item; take the nearest nprobe of those."""
    part_has = idx.partitions_with(mask)         # partition ids that contain a match
    if len(part_has) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    d = idx.cnorm[part_has] - 2.0 * (idx.centroids[part_has] @ q)
    take = part_has[np.argsort(d)[:nprobe]]
    return idx.search_parts(q, k, take, mask)