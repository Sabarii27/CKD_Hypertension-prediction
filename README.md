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
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows (Command Prompt)
.\venv\Scripts\Activate.ps1     # Windows (PowerShell)
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you don't have `requirements.txt`, create it with:

```
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
```

### 4. Download the trained model

The trained model (`end_to_end_cnn.h5`) is not included in the repository because of its size.  
You can download it from [this link](insert your Google Drive / GitHub release link).  
Place it in the `models/` folder.

Or train it yourself (see “Training from Scratch” below).

### 5. (Optional) Download the datasets

If you want to retrain the model or extract features, download:

- **AOD dataset** from Kaggle
- **RFMiD dataset** from Kaggle

Extract them into `data/AOD/` and `data/RFMiD/` respectively. Then run:

```bash
python prepare_multilabel_data.py   # Combines and preprocesses
python extract_features_multilabel.py   # Extracts vascular features (takes ~2‑3 hours)
python train_multilabel_classifier.py   # Trains the classifier
```

---

## 🖥️ Running the Web App

Once you have the trained model in `models/`, simply run:

```bash
streamlit run app.py
```

The app will open in your browser.

- **Home:** Overview and example image.
- **About:** Project details.
- **Prediction:** Enter user details, upload a retinal image, view probabilities, generate saliency maps, and download a PDF report.

---

## 🧪 Testing with a Single Image (Command Line)

```bash
python predict_cnn.py path/to/retinal_image.jpg
```

It will print predicted probabilities for all 8 diseases.

---

## 📂 Project Structure

```
CKD_Hypertension-prediction/
├── app.py                  # Streamlit web application
├── gradcam.py              # Saliency map function
├── config.py               # Path and image size configuration
├── generate_plots.py       # Generate ROC curves & confusion matrices
├── predict_cnn.py          # Command‑line prediction script
├── requirements.txt        # Python dependencies
├── models/                 # Trained model (end_to_end_cnn.h5)
├── results/                # Evaluation graphs and PDF reports
├── sample_images/          # (Optional) small set of test images
└── README.md               # This file
```

Other scripts (`prepare_multilabel_data.py`, `extract_features_multilabel.py`, `train_multilabel_classifier.py`) are used for training from scratch.

---

## 🛠️ Training from Scratch (If you have the datasets)

Download AOD and RFMiD from Kaggle (≈11 GB total).

Place them in `data/AOD/` and `data/RFMiD/`.

Run the preparation and training pipeline:

```bash
python prepare_multilabel_data.py
python extract_features_multilabel.py   # ~2‑3 hours
python train_multilabel_classifier.py
```

The trained model will be saved as `models/end_to_end_cnn.h5`.

---

## 📝 Future Improvements

- **Better vessel segmentation** – Replace the Frangi filter with a dedicated U‑Net trained on DRIVE/CHASE.
- **Multi‑modal prediction** – Add clinical metadata (age, blood pressure, eGFR) to boost accuracy.
- **Deploy online** – Host the Streamlit app on Hugging Face Spaces or Streamlit Cloud.
- **Mobile app** – Convert the model to TensorFlow Lite for offline smartphone screening.

---

## 🤝 Contributing

Feel free to open issues or submit pull requests. For major changes, please discuss them first.

---

## 📄 License

This project is for academic and research purposes. Please cite appropriately if you use it.

---

## 👨‍💻 Authors

- Ravendran M.
- Sabarinathan M.
- Sanjai M.

Under the guidance of Mrs. J. Jayapradha, Assistant Professor / CSE.

---

## 🙏 Acknowledgements

- Kaggle for providing the AOD and RFMiD datasets.
- The research community for public fundus image collections.
- TensorFlow, Streamlit, and other open‑source libraries.


