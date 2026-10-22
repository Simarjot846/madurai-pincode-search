import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import load
from ivf import IVFIndex
from workload import build_workload
from calibrate import calibrate

vecs, meta, cats, titles = load()
idx = IVFIndex(nlist=200).build(vecs)
idx.attach_original(vecs)
work = build_workload(vecs, meta, len(cats), 30)
calibrate(idx, meta, work)