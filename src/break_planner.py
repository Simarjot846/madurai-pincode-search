"""Step 8: break the planner on purpose, to show which part of the design matters."""
import time
import numpy as np
from filters import Predicate, exact_mask
from strategies import exact_scan
from planner import Planner
from workload import BUCKETS


class BrokenPlanner(Planner):
    def __init__(self, idx, meta, mode, **kw):
        super().__init__(idx, meta, **kw)
        self.mode = mode

    def estimate(self, preds):
        if self.mode == "indep_only":          # no sample correction
            return self.indep.estimate(preds), "indep"
        return super().estimate(preds)

    def choose(self, preds):
        sel, src = self.estimate(preds)
        small = sel * self.n <= self.t_exact_rows
        if self.mode == "always_post":
            return "postfilter_adaptive", sel, src
        if self.mode == "no_exact":
            small = False
        if self.mode == "inverted":
            if small:
                return "postfilter_adaptive", sel, src
            return "exact_scan", sel, src
        if small:
            return "exact_scan", sel, src
        if sel >= self.t_post_sel:
            return "postfilter_adaptive", sel, src
        return "partition_restricted", sel, src


def compare_modes(idx, meta, work, planners, k=10, n_filters=4, n_q=8):
    """planners: {name: planner}. Prints recall and p50 per bucket per planner."""
    print(f"{'bucket':8s} {'planner':12s} {'recall':>7s} {'p50ms':>8s}")
    for b in BUCKETS:
        for name, pl in planners.items():
            rec, lat = [], []
            for preds, qs in work[b][:n_filters]:
                mask = exact_mask(meta, preds)
                pl.search(qs[0], k, preds)                      # warm-up
                for q in qs[:n_q]:
                    truth = set(exact_scan(idx, q, k, mask)[0].tolist())
                    if not truth:
                        continue
                    t = time.perf_counter()
                    ids, _, _ = pl.search(q, k, preds)
                    lat.append((time.perf_counter() - t) * 1000)
                    rec.append(len(truth & set(ids.tolist())) / len(truth))
            print(f"{b:8s} {name:12s} {np.mean(rec):7.3f} {np.median(lat):8.2f}")
        print()


def worst_correlated_pairs(planner, meta, n_cats, n_zones, top=5):
    """Find category+zone pairs where the independence estimate is most wrong."""
    n = len(meta["price"])
    rows = []
    for c in range(n_cats):
        for z in range(n_zones):
            preds = [Predicate("category", "==", c), Predicate("zone", "==", z)]
            true = exact_mask(meta, preds).sum() / n
            est = planner.indep.estimate(preds)
            if true > 0:
                rows.append((est / true, c, z, est, true))
    rows.sort()
    print("independence UNDER-estimates (est/true smallest):")
    for r, c, z, est, true in rows[:top]:
        print(f"  cat={c:2d} zone={z:2d} true={true*100:6.3f}% indep_est={est*100:6.3f}%  ratio={r:.2f}")
    print("independence OVER-estimates (est/true largest):")
    for r, c, z, est, true in rows[-top:][::-1]:
        print(f"  cat={c:2d} zone={z:2d} true={true*100:6.3f}% indep_est={est*100:6.3f}%  ratio={r:.2f}")