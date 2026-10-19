import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import SteelDataset
from model import build_model
from thop import profile

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating using device: {device}")

    # 1. Load splits and test dataset
    df = pd.read_csv("splits.csv")
    test_df = df[df.split == "test"].reset_index(drop=True)
    test_loader = DataLoader(SteelDataset(test_df, is_train=False), batch_size=32, shuffle=False)

    # 2. Load trained model weights
    model = build_model("attentive_mobilenetv3", num_classes=6)
    weight_path = "outputs/attentive_mobilenetv3_best.pt"
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Missing {weight_path}! You will need to retrain if this file is missing.")
        
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Calculate Parameters and FLOPs
    dummy_input = torch.randn(1, 1, 224, 224).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"\n--- Model Parameters & FLOPs ---")
    print(f"Parameters: {params / 1e6:.2f} Million")
    print(f"FLOPs:      {flops / 1e9:.3f} GFLOPs")

    # 4. Measure Inference Latency (CPU)
    cpu_model = model.to("cpu").eval()
    cpu_dummy = torch.randn(1, 1, 224, 224)
    
    # Warmup
    for _ in range(10):
        _ = cpu_model(cpu_dummy)
        
    times = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            _ = cpu_model(cpu_dummy)
        times.append((time.perf_counter() - start) * 1000)
        
    mean_latency = np.mean(times)
    print(f"CPU Latency: {mean_latency:.2f} ms/image ({1000/mean_latency:.1f} FPS)")

    # 5. Evaluate Test Set Metrics
    model.to(device)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            p = model(x.to(device)).argmax(1).cpu().tolist()
            all_preds.extend(p)
            all_targets.extend(y.tolist())

    classes = sorted(df['label'].unique())
    print("\n--- Test Classification Report ---")
    print(classification_report(all_targets, all_preds, target_names=classes, digits=4))

    # 6. Save Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('NEU-DET Test Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    print("Saved confusion matrix plot to outputs/confusion_matrix.png")

    # 7. Export to ONNX Deployment Format
    # 7. Export to ONNX Deployment Format
    onnx_path = "outputs/attentive_mobilenetv3.onnx"
    torch.onnx.export(
        cpu_model,
        cpu_dummy,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamo=False
    )
    print(f"Successfully exported ONNX model to {onnx_path}")

if __name__ == "__main__":
    evaluate()