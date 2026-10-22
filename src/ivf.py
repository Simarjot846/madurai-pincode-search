"""Own IVF index: k-means partitions + inverted lists, search over the nearest nprobe partitions."""
import numpy as np
from sklearn.cluster import KMeans


class IVFIndex:
    def attach_original(self, vecs):

        self.orig_vecs = vecs
        self.part_of = np.empty(len(self.ids), dtype=np.int32)
        for p in range(self.nlist):
            self.part_of[self.ids[self.offsets[p]:self.offsets[p + 1]]] = p

    def partitions_with(self, mask):
        """Partition ids that contain at least one row where mask is True."""
        return np.unique(self.part_of[mask])

    def __init__(self, nlist=200, seed=42):
        self.nlist = nlist
        self.seed = seed

    def build(self, vecs):
        km = KMeans(n_clusters=self.nlist, n_init=1, max_iter=50, random_state=self.seed)
        assign = km.fit_predict(vecs)
        self.centroids = km.cluster_centers_.astype(np.float32)
        self.cnorm = (self.centroids ** 2).sum(axis=1)

        order = np.argsort(assign, kind="stable")
        self.ids = order.astype(np.int64)        # list position -> original row id
        self.vecs = np.ascontiguousarray(vecs[order])
        counts = np.bincount(assign, minlength=self.nlist)
        self.offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return self

    def probe(self, q, nprobe):
        """Partition ids sorted by distance of their centroid to the query."""
        d = self.cnorm - 2.0 * (self.centroids @ q)
        nprobe = min(nprobe, self.nlist)
        part = np.argpartition(d, nprobe - 1)[:nprobe]
        return part[np.argsort(d[part])]

    def candidates(self, parts):
        """List positions belonging to the given partitions."""
        segs = [np.arange(self.offsets[p], self.offsets[p + 1]) for p in parts]
        return np.concatenate(segs) if segs else np.empty(0, dtype=np.int64)

    def search_parts(self, q, k, parts, mask=None):
        """Top-k inside the given partitions. mask is a bool array over original row ids."""
        pos = self.candidates(parts)
        if mask is not None:
            pos = pos[mask[self.ids[pos]]]
        if len(pos) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        sims = self.vecs[pos] @ q
        kk = min(k, len(pos))
        top = np.argpartition(-sims, kk - 1)[:kk]
        top = top[np.argsort(-sims[top])]
        return self.ids[pos[top]], sims[top]

    def search(self, q, k, nprobe, mask=None):
        return self.search_parts(q, k, self.probe(q, nprobe), mask)

    def sizes(self):
        return np.diff(self.offsets)