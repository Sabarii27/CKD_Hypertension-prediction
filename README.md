# Retinal Image‑Based Prediction of Hypertension and Chronic Kidney Disease (CKD)

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red.svg)](https://streamlit.io/)

## 📌 Overview

This project uses **deep learning** to predict **hypertension** and **chronic kidney disease (CKD)** from retinal fundus images.  
A **ResNet50‑based multi‑label classifier** is trained on a combined dataset of over 16,000 retinal images. The system includes an interactive **Streamlit web app** with **saliency maps** (explainable AI) that highlight the image regions influencing each prediction.

**Key results:**
- **CKD:** Accuracy 94.6%, AUC 0.94
- **Hypertension:** Accuracy 99.9%, AUC 0.99

The tool is designed for non‑invasive, low‑cost screening – ideal for telemedicine and resource‑limited settings.

---

## ✨ Features

- **End‑to‑end deep learning pipeline** – from raw retinal images to disease probabilities.
- **Multi‑label classification** – predicts 8 conditions (CKD, hypertension, diabetes, AMD, glaucoma, cataract, myopia, normal).
- **Explainable AI** – saliency maps show which pixels influenced the prediction.
- **Interactive web app** – built with Streamlit; users upload an image, get probabilities, and generate heatmaps.
- **Downloadable PDF report** – includes user details, prediction percentages, and the heatmap for the higher‑risk disease.
- **Offline capable** – runs entirely on your local machine after setup.

---

## 📊 Dataset

We combined two public datasets:

| Dataset | Images | Disease categories |
|---------|--------|--------------------|
| **AOD** (Augmented Ocular Disease) | 14,800 | 7 (including hypertension) |
| **RFMiD 2.0** | 3,200 | 46 (including CKD‑related labels) |

**Total after cleaning:** 16,733 images with multi‑hot labels for 8 diseases.  
Preprocessing: green channel extraction, CLAHE, resizing to 224×224, normalization.

---

## 🧠 Model Architecture

- **Backbone:** ResNet50 pre‑trained on ImageNet.
- **Custom head:** GlobalAveragePooling → Dense(256, ReLU) → Dropout(0.3) → 8 parallel Dense(1, sigmoid) outputs.
- **Training strategy:**
  - Phase 1 (frozen base): 10 epochs, LR = 1e‑4, Adam.
  - Phase 2 (fine‑tuning): 10 epochs, LR = 1e‑5.
- **Loss:** Binary cross‑entropy per output.

---

## 📈 Results (Test Set)

| Disease       | Accuracy | AUC   | Precision | Recall | F1    |
|---------------|----------|-------|-----------|--------|-------|
| CKD           | 94.6%    | 0.94  | 93.2%     | 95.1%  | 94.1% |
| Hypertension  | 99.9%    | 0.99  | 99.8%     | 99.9%  | 99.8% |

ROC curves and confusion matrices are available in the `results/` folder.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10** (recommended; 3.11+ may cause compatibility issues)
- **Git** (to clone the repository)
- **8+ GB of RAM** (for feature extraction)
- **Optional:** NVIDIA GPU (for training) – but CPU is fine for inference.

### 1. Clone the repository

```bash
git clone https://github.com/Sabarii27/CKD_Hypertension-prediction.git
cd CKD_Hypertension-prediction