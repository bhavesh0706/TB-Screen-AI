from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "manifest_labeled.csv"
OUT = ROOT / "data" / "manifests"

df = pd.read_csv(MANIFEST)

print("Original distribution:")
print(df["label"].value_counts())

# Keep only binary classes
df = df[df["label"].isin(["Normal", "Tuberculosis"])].copy()

# 80% train, 10% val, 10% test
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42,
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42,
)

train_df.to_csv(OUT / "train.csv", index=False)
val_df.to_csv(OUT / "val.csv", index=False)
test_df.to_csv(OUT / "test.csv", index=False)

print("\nNew split distribution")

print("\nTRAIN")
print(train_df["label"].value_counts())

print("\nVAL")
print(val_df["label"].value_counts())

print("\nTEST")
print(test_df["label"].value_counts())