import pandas as pd

df = pd.read_csv("data/datasets/trifecta_train_features.csv")

print(df[["date", "venue", "race_no"]].head(20))
print(df["venue"].astype(str).str.strip().value_counts().head(30))
