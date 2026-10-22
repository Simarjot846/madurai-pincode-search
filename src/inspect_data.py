import pandas as pd

df = pd.read_csv("data/train.csv")

print("shape:", df.shape)
print("columns:", list(df.columns))
print()
print(df.head(3).T)
print()
print("nulls:\n", df.isna().sum())
print()
print("n categories:", df["categories"].nunique())
print(df["categories"].value_counts().head(30))
print("title lengths:", df["title"].str.len().describe())