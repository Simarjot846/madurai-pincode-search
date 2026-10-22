"""Query planner: picks one of three strategies from a scan-free selectivity estimate."""
import numpy as np
from filters import SelectivityEstimator, SampleEstimator, exact_mask
from strategies import exact_scan, postfilter_adaptive, partition_restricted

STRATEGIES = {
    "exact_scan": exact_scan,
    "postfilter_adaptive": postfilter_adaptive,
    "partition_restricted": partition_restricted,
}


class Planner:
    """
    Decision rule (thresholds are set from benchmark measurements, see calibrate in Step 6):
      est_rows <= t_exact_rows      -> exact_scan
      est_sel  >= t_post_sel        -> postfilter_adaptive
      otherwise                     -> partition_restricted
    Selectivity: sample estimate if the sample has >= min_hits matches, else the
    counter (independence) estimate capped at what the sample could not have missed.
    """

    def __init__(self, idx, meta, t_exact_rows=2500, t_post_sel=0.05, min_hits=5):
        self.idx, self.meta = idx, meta
        self.n = len(meta["price"])
        self.indep = SelectivityEstimator(meta)
        self.sample = SampleEstimator(meta)
        self.t_exact_rows = t_exact_rows
        self.t_post_sel = t_post_sel
        self.min_hits = min_hits

    def estimate(self, preds):
        sel_ind = self.indep.estimate(preds)
        sel_smp, hits = self.sample.estimate(preds)
        if hits >= self.min_hits:
            return sel_smp, "sample"
        cap = self.min_hits / len(self.sample.idx)
        return min(sel_ind, cap), "counters"

    def choose(self, preds):
        sel, src = self.estimate(preds)
        if sel * self.n <= self.t_exact_rows:
            return "exact_scan", sel, src
        if sel >= self.t_post_sel:
            return "postfilter_adaptive", sel, src
        return "partition_restricted", sel, src

    def search(self, q, k, preds, force=None):
        strat, sel, src = self.choose(preds)
        if force:
            strat = force
        mask = exact_mask(self.meta, preds)      # mask build counts toward latency
        ids, sims = STRATEGIES[strat](self.idx, q, k, mask)
        return ids, sims, strat