from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

records = []

# -----------------------------------
# Shenzhen
# -----------------------------------

shenzhen = pd.read_csv(RAW / "Shenzhen" / "shenzhen_metadata.csv")

for _, row in shenzhen.iterrows():
    label = "Normal" if row["findings"].lower() == "normal" else "Tuberculosis"

    records.append(
        {"dataset": "Shenzhen", "filename": row["study_id"], "label": label}
    )

# -----------------------------------
# Montgomery
# -----------------------------------

mont = pd.read_csv(RAW / "Montgomery" / "montgomery_metadata.csv")

for _, row in mont.iterrows():
    label = "Normal" if row["findings"].lower() == "normal" else "Tuberculosis"

    records.append(
        {"dataset": "Montgomery", "filename": row["study_id"], "label": label}
    )

# -----------------------------------
# TBX11K (folder labels)
# -----------------------------------

tbx_root = RAW / "TBX11K" / "imgs"

# health -> Normal
for img in (tbx_root / "health").glob("*.png"):
    records.append({
        "dataset": "TBX11K",
        "filename": img.name,
        "label": "Normal"
    })

# sick -> Tuberculosis
for img in (tbx_root / "sick").glob("*.png"):
    records.append({
        "dataset": "TBX11K",
        "filename": img.name,
        "label": "Tuberculosis"
    })

health_count = len(list((tbx_root / "health").glob("*.png")))
sick_count = len(list((tbx_root / "sick").glob("*.png")))

print(f"TBX11K Health: {health_count} | TBX11K Sick: {sick_count}")

# -----------------------------------
# Save Manifest
# -----------------------------------

out_dir = ROOT / "data" / "manifests"
out_dir.mkdir(parents=True, exist_ok=True)

manifest = pd.DataFrame(records)
out = out_dir / "manifest_labeled.csv"
manifest.to_csv(out, index=False)

print("\nLabel Summary")
print(manifest["label"].value_counts())

print("\nDataset Summary")
print(manifest["dataset"].value_counts())

print("\nSaved:", out)