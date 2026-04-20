import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- AUTO-LOCATE DATA ----------
base_dir = 'data'
found = False
for root, dirs, files in os.walk(base_dir):
    if 'multilabel.csv' in files:
        labels_csv = os.path.join(root, 'multilabel.csv')
        images_dir = os.path.join(root, 'images')
        if os.path.isdir(images_dir):
            found = True
            break
if not found:
    raise FileNotFoundError("Could not find multilabel.csv and images folder under 'data'.")

print(f"Found labels at: {labels_csv}")
print(f"Found images at: {images_dir}")

model_path = 'models/end_to_end_cnn.h5'
IMG_SIZE = 224
BATCH_SIZE = 32
target_columns = ['ckd', 'hypertension', 'diabetes', 'amd', 'glaucoma', 'cataract', 'myopia', 'normal']

# ---------- LOAD DATA ----------
df = pd.read_csv(labels_csv)

# Build list of existing image paths and keep valid rows
image_paths = []
valid_indices = []
for idx, row in df.iterrows():
    img_id = row['image_id']
    for ext in ['.jpg', '.png']:
        path = os.path.join(images_dir, img_id + ext)
        if os.path.exists(path):
            image_paths.append(path)
            valid_indices.append(idx)
            break
df = df.iloc[valid_indices].reset_index(drop=True)
labels = df[target_columns].values.astype(np.float32)
print(f"Total valid images: {len(image_paths)}")

# ---------- CREATE COMPOSITE LABEL (CKD OR HYPERTENSION) ----------
composite = (labels[:, 0] + labels[:, 1]) > 0
composite = composite.astype(int)
print(f"Composite label distribution: {np.bincount(composite)}")

X = image_paths
y = labels

# ---------- SEARCH FOR A RANDOM STATE THAT GIVES POSITIVES FOR BOTH ----------
ckd_pos_total = np.sum(y[:, 0])
htn_pos_total = np.sum(y[:, 1])
print(f"Total CKD positives: {ckd_pos_total}")
print(f"Total Hypertension positives: {htn_pos_total}")

best_state = None
ckd_test = 0
htn_test = 0
for seed in range(1, 100):
    # First split: train_val (85%) and test (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=composite
    )
    ckd_test = np.sum(y_test[:, 0])
    htn_test = np.sum(y_test[:, 1])
    if ckd_test > 0 and htn_test > 0:
        best_state = seed
        break

if best_state is None:
    # If no seed found, try a larger range or fallback to a simple split
    best_state = 42
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=best_state, stratify=composite
    )
    print("Could not find a seed giving both positives; using seed=42.")
else:
    print(f"Found seed {best_state} with CKD test positives: {ckd_test}, HTN test positives: {htn_test}")

# Now split train_val into train (70% of total) and val (15% of total)
train_ratio = 0.7 / 0.85
composite_train_val = (y_train_val[:, 0] + y_train_val[:, 1]) > 0
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=1 - train_ratio, random_state=best_state,
    stratify=composite_train_val
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"CKD positives in test: {np.sum(y_test[:, 0])}")
print(f"Hypertension positives in test: {np.sum(y_test[:, 1])}")

# ---------- CREATE TENSORFLOW DATASETS ----------
def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    label_tuple = tf.unstack(tf.cast(label, tf.float32), axis=0)
    return image, tuple(label_tuple)

# Build test dataset directly from X_test and y_test
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
print(f"Test set size: {len(X_test)} images")

# ---------- LOAD MODEL ----------
model = load_model(model_path)
print("Model loaded.")

# ---------- COLLECT PREDICTIONS ----------
y_true = []
y_pred_prob = []
for images, labels_tuple in test_dataset:
    labels_batch = tf.stack(labels_tuple, axis=1).numpy()
    y_true.append(labels_batch)
    preds = model.predict(images, verbose=0)
    prob_batch = np.concatenate(preds, axis=1)
    y_pred_prob.append(prob_batch)

y_true = np.vstack(y_true)
y_pred_prob = np.vstack(y_pred_prob)
print(f"Predictions shape: {y_pred_prob.shape}, True labels shape: {y_true.shape}")

# ---------- SAVE ROC CURVES (ALL) ----------
plt.figure(figsize=(10, 8))
for i, disease in enumerate(target_columns):
    fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{disease} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves – All Diseases')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curves_all.png', dpi=150)
plt.close()
print("Saved: roc_curves_all.png")

# ---------- SAVE ROC CURVES (CKD & HYPERTENSION) ----------
plt.figure(figsize=(8, 6))
for disease, idx in [('CKD', 0), ('Hypertension', 1)]:
    fpr, tpr, _ = roc_curve(y_true[:, idx], y_pred_prob[:, idx])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{disease} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves – CKD & Hypertension')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curves_ckd_htn.png', dpi=150)
plt.close()
print("Saved: roc_curves_ckd_htn.png")

# ---------- SAVE CONFUSION MATRICES (ALL) ----------
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()
for i, disease in enumerate(target_columns):
    y_pred_binary = (y_pred_prob[:, i] >= 0.5).astype(int)
    cm = confusion_matrix(y_true[:, i], y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
    axes[i].set_title(disease)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrices_all.png', dpi=150)
plt.close()
print("Saved: confusion_matrices_all.png")

# ---------- SAVE CONFUSION MATRICES (CKD & HYPERTENSION) ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for idx, disease in enumerate(['CKD', 'Hypertension']):
    y_pred_binary = (y_pred_prob[:, idx] >= 0.5).astype(int)
    cm = confusion_matrix(y_true[:, idx], y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
    axes[idx].set_title(disease)
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrices_ckd_htn.png', dpi=150)
plt.close()
print("Saved: confusion_matrices_ckd_htn.png")

print("All plots generated successfully.")