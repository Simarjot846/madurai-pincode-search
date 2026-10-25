"""Data preparation: embeds product titles and attaches category / price / zone metadata."""
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

SEED = 42
rng = np.random.default_rng(SEED)

ZONES = [
    "Mumbai-MH", "Delhi-NCR", "Bengaluru-KA", "Hyderabad-TS", "Kolkata-WB",
    "Chennai-TN", "Pune-MH", "Ahmedabad-GJ", "Jaipur-RJ", "Lucknow-UP",
    "Kochi-KL", "Chandigarh-PB", "Bhopal-MP", "Patna-BR", "Nagpur-MH",
    "Indore-MP", "Coimbatore-TN", "Visakhapatnam-AP", "Bhubaneswar-OD",
    "Ranchi-JH", "Dehradun-UK", "Guwahati-AS", "Madurai-TN", "Raipur-CG",
    "Goa-GA", "Srinagar-JK", "Imphal-MN", "Gangtok-SK", "PortBlair-AN", "Leh-LA",
]
N_BIG = 26          
TAIL_W = 0.0005     
PRICE_MEDIAN = {
    "Arts, Crafts & Sewing": 450, "Beauty": 500, "All Beauty": 450,
    "Sports & Outdoors": 1200, "Grocery & Gourmet Food": 300,
    "Cell Phones & Accessories": 900, "Health & Personal Care": 400,
    "Tools & Home Improvement": 1100, "Clothing, Shoes & Jewelry": 1300,
    "Baby": 700, "Baby Products": 750, "Patio, Lawn & Garden": 1500,
    "Electronics": 3500, "All Electronics": 3200, "Automotive": 1600,
    "Toys & Games": 800, "Office Products": 600, "Appliances": 5500,
    "Musical Instruments": 4500, "Industrial & Scientific": 2000,
    "Pet Supplies": 650,
}

FIXED_HOME = {
    "Clothing, Shoes & Jewelry": [22],     
    "Grocery & Gourmet Food": [10, 16],     
    "Electronics": [2],                     
}


def zone_base_weights():
    w = 1.0 / (np.arange(1, N_BIG + 1) ** 1.2)
    w = w / w.sum() * (1 - 4 * TAIL_W)
    return np.concatenate([w, np.full(4, TAIL_W)])


def assign_zones(categories):
    base = zone_base_weights()
    zones = np.empty(len(categories), dtype=np.int16)
    for c in np.unique(categories):
        idx = np.where(categories == c)[0]
        home = FIXED_HOME.get(c)
        if home is None:
            home = rng.choice(np.arange(3, N_BIG), size=2, replace=False).tolist()
        p = 0.55 * base
        for h in home:
            p[h] += 0.45 / len(home)
        p = p / p.sum()
        zones[idx] = rng.choice(len(ZONES), size=len(idx), p=p)
    return zones


def main():
    os.makedirs("data/processed", exist_ok=True)
    df = pd.read_csv("data/train.csv")
    df = df.dropna(subset=["title"]).drop_duplicates(subset="ImgId").reset_index(drop=True)
    print("rows after cleaning:", len(df))

    cats = df["categories"].to_numpy()
    zone = assign_zones(cats)
    med = df["categories"].map(PRICE_MEDIAN).to_numpy()
    price = np.clip(rng.lognormal(np.log(med), 0.8), 49, 99999).round().astype(np.int32)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model.max_seq_length = 64
    vecs = model.encode(
        df["title"].tolist(), batch_size=128,
        normalize_embeddings=True, show_progress_bar=True,
    ).astype(np.float32)

    meta = pd.DataFrame({
        "id": df["ImgId"], "title": df["title"],
        "category": cats, "price": price, "zone": zone,
    })
    np.save("data/processed/vectors.npy", vecs)
    meta.to_csv("data/processed/meta.csv", index=False)

    print("vectors:", vecs.shape)
    print("\nitems per zone (share %):")
    share = meta["zone"].value_counts(normalize=True).sort_index() * 100
    for z, s in share.items():
        print(f"  {z:2d} {ZONES[z]:18s} {s:6.3f}%  ({(meta['zone'] == z).sum()})")
    print("\nClothing share by zone (top 3):")
    cl = meta[meta.category == "Clothing, Shoes & Jewelry"]["zone"].value_counts(normalize=True).head(3)
    print(cl.rename(index=lambda z: ZONES[z]).round(3).to_string())
    print("\nprice (Rs) describe:\n", meta["price"].describe().round(0).to_string())


if __name__ == "__main__":
    main()