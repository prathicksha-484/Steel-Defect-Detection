import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from dataset import SteelDataset
from model import build_model

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    df = pd.read_csv("splits.csv")

    train_loader = DataLoader(SteelDataset(df[df.split=="train"], is_train=True), batch_size=32, shuffle=True)
    val_loader = DataLoader(SteelDataset(df[df.split=="val"], is_train=False), batch_size=64)

    model = build_model("attentive_mobilenetv3", num_classes=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    best_f1 = 0.0
    for epoch in range(15):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                p = model(x.to(device)).argmax(1).cpu().tolist()
                preds.extend(p)
                targets.extend(y.tolist())

        val_f1 = f1_score(targets, preds, average="macro")
        print(f"Epoch {epoch+1:02d} | Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "outputs/attentive_mobilenetv3_best.pt")
            print("  --> Best model saved to outputs/")

if __name__ == "__main__":
    train()