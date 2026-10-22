"""Filter predicates, exact masks, and a scan-free selectivity estimator."""
import numpy as np


class Predicate:
    """One predicate: ('category', '==', value), ('zone', '==', value) or ('price', 'between', (lo, hi))."""

    def __init__(self, field, op, value):
        assert field in ("category", "zone", "price")
        assert op in ("==", "between")
        self.field, self.op, self.value = field, op, value

    def __repr__(self):
        return f"{self.field} {self.op} {self.value}"


def exact_mask(meta, preds):
    """Boolean mask over all items. meta is a dict of numpy arrays: category, zone, price."""
    n = len(meta["price"])
    mask = np.ones(n, dtype=bool)
    for p in preds:
        col = meta[p.field]
        if p.op == "==":
            mask &= (col == p.value)
        else:
            lo, hi = p.value
            mask &= (col >= lo) & (col <= hi)
    return mask


class SelectivityEstimator:
    """Built once at index time from counters and a price histogram. Never touches the data at query time."""

    def __init__(self, meta, n_price_bins=50):
        self.n = len(meta["price"])
        self.cat_count = np.bincount(meta["category"])
        self.zone_count = np.bincount(meta["zone"])
        qs = np.linspace(0, 1, n_price_bins + 1)
        self.price_edges = np.unique(np.quantile(meta["price"], qs))

    def _price_frac(self, lo, hi):
        # fraction of items with price in [lo, hi], assuming uniform inside each equal-frequency bin
        e = self.price_edges
        nb = len(e) - 1
        total = 0.0
        for i in range(nb):
            b_lo, b_hi = e[i], e[i + 1]
            width = max(b_hi - b_lo, 1e-9)
            overlap = max(0.0, min(hi, b_hi) - max(lo, b_lo))
            total += (overlap / width) / nb
        return min(total, 1.0)

    def estimate(self, preds):
        """Returns estimated fraction (0..1) of items matching, assuming independent predicates."""
        sel = 1.0
        for p in preds:
            if p.field == "category":
                sel *= self.cat_count[p.value] / self.n
            elif p.field == "zone":
                sel *= self.zone_count[p.value] / self.n
            else:
                lo, hi = p.value
                sel *= self._price_frac(lo, hi)
        return sel
    
class SampleEstimator:
    """Estimates selectivity from a fixed random sample of rows. Handles correlated predicates."""

    def __init__(self, meta, size=2000, seed=7):
        rng = np.random.default_rng(seed)
        n = len(meta["price"])
        self.idx = rng.choice(n, size=min(size, n), replace=False)
        self.sample = {k: v[self.idx] for k, v in meta.items()}
        self.n = n

    def estimate(self, preds):
        m = exact_mask(self.sample, preds)
        hits = int(m.sum())
        return hits / len(self.idx), hits