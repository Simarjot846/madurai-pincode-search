import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dataset import load
from filters import Predicate, exact_mask, SelectivityEstimator

vecs, meta, cats, titles = load()
est = SelectivityEstimator(meta)
C = {c: i for i, c in enumerate(cats)}

tests = {
    "zone=Mumbai":               [Predicate("zone", "==", 0)],
    "cat=Pet Supplies":          [Predicate("category", "==", C["Pet Supplies"])],
    "price 500-1500":            [Predicate("price", "between", (500, 1500))],
    "cat=Clothing & zone=Madurai": [Predicate("category", "==", C["Clothing, Shoes & Jewelry"]), Predicate("zone", "==", 22)],
    "Clothing & Madurai & price<1500": [Predicate("category", "==", C["Clothing, Shoes & Jewelry"]), Predicate("zone", "==", 22), Predicate("price", "between", (49, 1500))],
    "cat=Pet & zone=Madurai":    [Predicate("category", "==", C["Pet Supplies"]), Predicate("zone", "==", 22)],
    "zone=Leh":                  [Predicate("zone", "==", 29)],
    "Electronics & Leh":         [Predicate("category", "==", C["Electronics"]), Predicate("zone", "==", 29)],
}

print(f"{'filter':36s} {'true':>9s} {'estimate':>9s} {'ratio':>7s}")
for name, preds in tests.items():
    true = exact_mask(meta, preds).mean()
    e = est.estimate(preds)
    ratio = e / true if true > 0 else float("inf")
    print(f"{name:36s} {true:9.5f} {e:9.5f} {ratio:7.2f}")