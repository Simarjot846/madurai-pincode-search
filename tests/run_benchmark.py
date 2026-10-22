import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import load
from ivf import IVFIndex
from workload import build_workload
from planner import Planner
from benchmark import run_all

vecs, meta, cats, titles = load()

idx = IVFIndex(nlist=200).build(vecs)
idx.attach_original(vecs)      # needed by pure_prefilter (idx.orig_vecs)
planner = Planner(idx, meta)

work = build_workload(vecs, meta, len(cats), 30)
run_all(planner, meta, work)