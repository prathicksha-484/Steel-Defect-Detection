import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import glob
import torch
import cv2
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from model import build_model

class GC10Dataset(Dataset):
    def __init__(self, image_paths, img_size=224):
        self.image_paths = image_paths
        self.transforms = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.5,), std=(0.25,)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback if image path issue occurs
            img = np.zeros((224, 224), dtype=np.uint8)
        augmented = self.transforms(image=img[..., None])
        return augmented['image'], path

def run_cross_eval():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Cross-Dataset Benchmark on device: {device}")

    # 1. Search for images in GC10-DET directory
    gc10_paths = glob.glob("GC10-DET/**/*.jpg", recursive=True) + \
                 glob.glob("GC10-DET/**/*.png", recursive=True) + \
                 glob.glob("GC10-DET/**/*.bmp", recursive=True)

    if not gc10_paths:
        print("Warning: No GC10-DET images found in GC10-DET/ folder.")
        print("Please ensure the GC10-DET dataset folder is extracted in the root directory.")
        return

    print(f"Found {len(gc10_paths)} images in GC10-DET.")

    # 2. Load model
    model = build_model("attentive_mobilenetv3", num_classes=6)
    weight_path = "outputs/attentive_mobilenetv3_best.pt"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device).eval()

    # 3. Create DataLoader
    dataset = GC10Dataset(gc10_paths)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    df_neu = pd.read_csv("splits.csv")
    classes = sorted(df_neu['label'].unique())

    # 4. Run Inference on Domain-Shifted Dataset
    predictions = []
    print("Running inference across GC10-DET dataset...")
    with torch.no_grad():
        for x, paths in loader:
            outputs = model(x.to(device))
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, dim=1)
            
            for path, pred_idx, conf in zip(paths, preds.cpu().tolist(), confidences.cpu().tolist()):
                predictions.append({
                    "path": path,
                    "predicted_class": classes[pred_idx],
                    "confidence": conf
                })

    results_df = pd.DataFrame(predictions)
    results_df.to_csv("outputs/gc10_predictions.csv", index=False)
    
    print("\n--- GC10-DET Prediction Distribution ---")
    print(results_df['predicted_class'].value_counts())
    print("\nMean Prediction Confidence:", f"{results_df['confidence'].mean() * 100:.2f}%")
    print("Saved prediction results to outputs/gc10_predictions.csv")

if __name__ == "__main__":
    run_cross_eval()