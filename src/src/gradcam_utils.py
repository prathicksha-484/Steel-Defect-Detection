import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from dataset import SteelDataset
from model import build_model

def generate_gradcam():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Generating Grad-CAM using device: {device}")

    # 1. Load splits dataframe
    df = pd.read_csv("splits.csv")
    test_df = df[df.split == "test"].reset_index(drop=True)

    # 2. Rebuild model and load trained weights
    model = build_model("attentive_mobilenetv3", num_classes=6)
    weight_path = "outputs/attentive_mobilenetv3_best.pt"
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Missing weight file at {weight_path}. Run src/train.py first!")

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device).eval()

    # 3. Target the CBAM attention layer for CAM visualization
    target_layers = [model.cbam]
    cam = GradCAM(model=model, target_layers=target_layers)

    # 4. Extract 1 representative sample image for each defect class
    classes = sorted(df['label'].unique())
    fig, axes = plt.subplots(len(classes), 2, figsize=(8, 14))

    for idx, label in enumerate(classes):
        sample = test_df[test_df.label == label].iloc[0]
        
        # Load grayscale image and resize to model dimensions (224x224)
        img_gray = cv2.imread(sample['path'], cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img_gray, (224, 224))

        # Normalize tensor input exactly as done during training
        input_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float() / 255.0
        input_tensor = (input_tensor - 0.5) / 0.25

        # Compute Grad-CAM activation heatmap
        grayscale_cam = cam(input_tensor=input_tensor.to(device), targets=None)[0, :]

        # Prepare 3-channel RGB image normalized to [0, 1] for heatmap overlay
        rgb_img = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB) / 255.0
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Plot Original Image
        axes[idx, 0].imshow(rgb_img)
        axes[idx, 0].set_title(f"Original: {label}", fontsize=10, fontweight='bold')
        axes[idx, 0].axis('off')

        # Plot Grad-CAM Heatmap Overlay
        axes[idx, 1].imshow(visualization)
        axes[idx, 1].set_title(f"Grad-CAM Heatmap: {label}", fontsize=10, fontweight='bold')
        axes[idx, 1].axis('off')

    plt.suptitle("MobileNetV3 + CBAM Grad-CAM Defect Localization", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    save_path = "outputs/gradcam_results.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved Grad-CAM plot to {save_path}")

    # Step Verification
    if os.path.exists(save_path):
        print("Verification SUCCESS: Grad-CAM heatmap grid generated.")

if __name__ == "__main__":
    generate_gradcam()