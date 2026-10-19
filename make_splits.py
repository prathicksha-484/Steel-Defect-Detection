import os
import glob
import hashlib
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

print("Scanning NEU-DET folder...")
ROOT = "NEU-DET"

image_paths = glob.glob(f"{ROOT}/**/*.jpg", recursive=True) + \
              glob.glob(f"{ROOT}/**/*.bmp", recursive=True) + \
              glob.glob(f"{ROOT}/**/*.png", recursive=True)

rows, hashes, bad = [], {}, []

for p in image_paths:
    if "annotations" in p.lower():
        continue
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            a = np.array(im)
            h = hashlib.md5(a.tobytes()).hexdigest()
            hashes.setdefault(h, []).append(p)
            label = os.path.basename(os.path.dirname(p))
            rows.append({"path": p, "label": label, "w": im.width, "h": im.height, "md5": h})
    except Exception as e:
        bad.append((p, str(e)))

df = pd.DataFrame(rows)
df = df.drop_duplicates(subset=["md5"]).reset_index(drop=True)

classes = sorted(df['label'].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
df["y"] = df['label'].map(class_to_idx)

train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['y'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['y'], random_state=42)

splits_df = pd.concat([
    train_df.assign(split="train"),
    val_df.assign(split="val"),
    test_df.assign(split="test")
], ignore_index=True)

splits_df.to_csv("splits.csv", index=False)
print("Created splits.csv successfully!")
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")