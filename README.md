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


### 2. Create and activate a virtual environment
Windows (Command Prompt):

bash
python -m venv venv
venv\Scripts\activate
Windows (PowerShell):

bash
python -m venv venv
.\venv\Scripts\Activate.ps1
Linux / Mac:

bash
python3 -m venv venv
source venv/bin/activate
### 3. Install required Python packages
bash
pip install -r requirements.txt
If you do not have requirements.txt, create it with the following content:

text
streamlit
tensorflow==2.10.0
opencv-python
numpy
matplotlib
scikit-learn
pandas
seaborn
tqdm
pillow
scikit-image
fpdf
Then run pip install -r requirements.txt.

4. Get the trained model
The trained model (end_to_end_cnn.h5) is not stored in this repository because it is too large.
You have two options:

Option A – Download the pre‑trained model (recommended)
Download the model from this link (insert your actual Google Drive / release link).

Place the file inside the models/ folder (create the folder if it doesn't exist).

Option B – Train the model from scratch
If you have the AOD and RFMiD datasets (see Training from Scratch below), you can train the model yourself. This will take several hours.

5. Run the web application
Once the model is in models/, start the Streamlit app:

bash
streamlit run app.py
Your browser will open automatically.

Go to the Prediction page, enter your details, upload a retinal image, and view the predictions.

You can also generate a saliency map (heatmap) and download a PDF report.

6. (Optional) Test a single image from the command line
bash
python predict_cnn.py path/to/retinal_image.jpg
This will print the probabilities for all 8 diseases.

⚙️ Training from scratch (if you have the datasets)
Download the AOD and RFMiD datasets from Kaggle (links above in the Dataset section).

Extract them into data/AOD/ and data/RFMiD/.

Run the following scripts in order:

bash
python prepare_multilabel_data.py         # Combines and preprocesses images
python extract_features_multilabel.py     # Extracts vascular features (takes 2‑3 hours)
python train_multilabel_classifier.py     # Trains the classifier (fast)
The trained model will be saved as models/end_to_end_cnn.h5.

