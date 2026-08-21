from pathlib import Path
from PIL import Image
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT/"configs/dataset.yaml"))

records=[]

sources={
    "TBX11K":ROOT/cfg["datasets"]["tbx11k"]/ "imgs",
    "Shenzhen":ROOT/cfg["datasets"]["shenzhen"]/ "images",
    "Montgomery":ROOT/cfg["datasets"]["montgomery"]/ "images",
    "LungSegmentation704":ROOT/cfg["datasets"]["lung_segmentation"]/ "image"
}

ext={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

for name,folder in sources.items():
    for f in folder.rglob("*"):
        if f.suffix.lower() not in ext:
            continue
        try:
            with Image.open(f) as img:
                w,h=img.size
            records.append([name,f.name,str(f.relative_to(ROOT)),w,h,"OK"])
        except:
            records.append([name,f.name,str(f.relative_to(ROOT)),None,None,"CORRUPTED"])

df=pd.DataFrame(records,columns=[
    "dataset","filename","path","width","height","status"
])

out=ROOT/cfg["output"]["manifest"]
out.parent.mkdir(parents=True,exist_ok=True)
df.to_csv(out,index=False)

print("\n=== DATASET SUMMARY ===")
print(df.groupby("dataset").size())
print("\nCorrupted:",(df.status=="CORRUPTED").sum())
print("\nSaved:",out)