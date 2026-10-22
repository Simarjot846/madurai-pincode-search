"""Loads vectors + metadata, encodes category names as integer codes."""
import numpy as np
import pandas as pd


def load(path_vecs="data/processed/vectors.npy", path_meta="data/processed/meta.csv"):
    vecs = np.load(path_vecs)
    df = pd.read_csv(path_meta)
    cat_names = sorted(df["category"].unique())
    cat_code = {c: i for i, c in enumerate(cat_names)}
    meta = {
        "category": df["category"].map(cat_code).to_numpy().astype(np.int16),
        "zone": df["zone"].to_numpy().astype(np.int16),
        "price": df["price"].to_numpy().astype(np.int32),
    }
    return vecs, meta, cat_names, df["title"].tolist()