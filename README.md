# Lightweight Steel Surface Defect Classification Pipeline
### Real-Time Industrial Inspection using MobileNetV3 and CBAM

An end-to-end, edge-ready deep learning pipeline designed to inspect steel surface defects at high speeds on factory conveyor belts. Built with **MobileNetV3-Small** and **Convolutional Block Attention Module (CBAM)**, achieving **100% Macro F1 accuracy** while remaining ultra-lightweight for CPU edge deployment.

---

## 🚀 Key Performance Metrics
* **Test Accuracy / Macro F1:** 100% (Evaluated on NEU-DET test split)
* **Model Parameters:** 3.19 Million (Extremely lightweight vs heavy 25M+ networks)
* **Computational Complexity:** 0.212 GFLOPs per frame
* **Inference Speed:** ~32.7 FPS on standard CPU (~30.6 ms execution time)
* **Deployment Format:** ONNX (Open Neural Network Exchange) ready for C++/TensorRT edge integration

---

## 🛠️ Project Architecture & Highlights
1. **Data Hygiene & Deduplication (`make_splits.py`):** Uses MD5 hash checks to eliminate duplicate samples across classes, ensuring zero data leakage before structuring into a clean 70/15/15 train/val/test split.
2. **MobileNetV3-Small Backbone:** Utilizes depthwise separable convolutions and inverted residual blocks to slash computational overhead by ~85%.
3. **CBAM Dual-Attention Mechanism:** 
   * **Channel Attention ("WHAT"):** Prioritizes defect-sensitive feature channels.
   * **Spatial Attention ("WHERE"):** Isolates precise defect coordinates, filtering out factory lighting glares and oil reflections.
4. **Explainability (Grad-CAM):** Generates colorized activation heatmaps overlayed on steel sheets to prove the model focuses on physical cracks rather than background artifacts.
5. **Cross-Dataset Generalization:** Evaluated against the **GC10-DET** dataset to prove resilience against unseen domain shifts in camera sensors and factory environments.

---

## 📂 Repository Structure

```text
steel-defect-detection/
│
├── NEU-DET/                # NEU Surface Defect Database
├── GC10-DET/               # Cross-validation dataset
├── outputs/                # Generated model weights, logs, and plots
│   ├── attentive_mobilenetv3_best.pt
│   ├── attentive_mobilenetv3.onnx
│   ├── confusion_matrix.png
│   ├── gradcam_results.png
│   └── gc10_predictions.csv
│
├── src/                    # Source code modules
│   ├── dataset.py          # Data loading & augmentation pipeline
│   ├── model.py            # MobileNetV3 + CBAM architecture definition
│   ├── train.py            # Training loop with AdamW & Cosine Annealing
│   ├── evaluate.py         # Test evaluation, metrics, and ONNX export
│   ├── gradcam_utils.py    # Grad-CAM heatmap generation
│   └── cross_eval.py       # GC10-DET cross-dataset validation
│
├── make_splits.py          # MD5 hash deduplication & data partitioning script
├── dashboard.html          # Interactive presentation & metrics dashboard
└── README.md               # Project documentation
