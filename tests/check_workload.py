import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import load
from workload import build_workload, BUCKETS
from filters import exact_mask

vecs, meta, cats, titles = load()
work = build_workload(vecs, meta, len(cats), 30)
for b in BUCKETS:
    print(f"{b:8s} filters={len(work[b])}")
    for preds, qs in work[b][:2]:
        print("    ", preds, "matches:", int(exact_mask(meta, preds).sum()))