from pathlib import Path
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

# Original manifest with correct paths
manifest = pd.read_csv(ROOT / "data" / "manifests" / "manifest.csv")

# Labeled manifest
labels = pd.read_csv(ROOT / "data" / "manifests" / "manifest_labeled.csv")

# Merge using dataset + filename
df = manifest.merge(labels, on=["dataset", "filename"], how="inner")

print(f"Merged records: {len(df)}")

# Remove old split
for split in ["train", "val", "test"]:
    folder = ROOT / "data" / split
    if folder.exists():
        shutil.rmtree(folder)

# Stratified split
train, temp = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42,
)

val, test = train_test_split(
    temp,
    test_size=0.50,
    stratify=temp["label"],
    random_state=42,
)

# Copy processed images with debug tracking
copied = 0
missing = 0

for split_name, split_df in {
    "train": train,
    "val": val,
    "test": test,
}.items():

    for _, row in split_df.iterrows():
        src = ROOT / "data" / "processed" / row["dataset"] / row["filename"]
        dst = ROOT / "data" / split_name / row["label"] / row["filename"]

        dst.parent.mkdir(parents=True, exist_ok=True)

        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1

print("\nReal dataset split created.")
print(f"Copied:  {copied}")
print(f"Missing: {missing}")

print(f"\nTrain: {len(train)}")
print(f"Val:   {len(val)}")
print(f"Test:  {len(test)}")