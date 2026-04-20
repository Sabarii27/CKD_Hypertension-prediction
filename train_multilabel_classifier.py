# train_multilabel_classifier.py
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from config import DATA_DIR, MODELS_DIR, RESULTS_DIR

# Load features and labels
features = pd.read_csv(os.path.join(DATA_DIR, 'combined', 'features.csv'))
labels = pd.read_csv(os.path.join(DATA_DIR, 'combined', 'multilabel.csv'))

# Merge on image_id
df = features.merge(labels, on='image_id', how='inner')
print(f"Total samples with features: {len(df)}")

# Define feature columns (all except image_id and the label columns)
feature_cols = [c for c in features.columns if c != 'image_id']
X = df[feature_cols].values

# Define target columns (disease labels we want to predict)
target_cols = ['ckd', 'hypertension', 'diabetes', 'amd', 'glaucoma', 'cataract', 'myopia', 'normal']
y = df[target_cols].values

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Train a multi‑output Random Forest
base_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
multi_clf = MultiOutputClassifier(base_rf)
multi_clf.fit(X_train, y_train)

# Save model
model_path = os.path.join(MODELS_DIR, 'multilabel_classifier.pkl')
joblib.dump(multi_clf, model_path)
print(f"Model saved to {model_path}")

# Predict on test set
y_pred = multi_clf.predict(X_test)

# Evaluate per disease
print("\n=== Per‑Disease Accuracy ===")
for i, disease in enumerate(target_cols):
    acc = accuracy_score(y_test[:, i], y_pred[:, i])
    print(f"{disease:12s}: {acc:.4f}")

# Overall exact match accuracy (all diseases correct)
exact_match = np.mean(np.all(y_pred == y_test, axis=1))
print(f"\nExact match accuracy (all diseases correct): {exact_match:.4f}")

# Generate ROC curves for each disease (if binary)
# We'll need probabilities for ROC; MultiOutputClassifier can give probabilities if the base estimator supports it.
# RandomForest supports predict_proba, so we can get probabilities.
y_proba = multi_clf.predict_proba(X_test)  # returns a list of arrays, one per output

plt.figure(figsize=(12, 8))
for i, disease in enumerate(target_cols):
    # Check if there are at least two classes in y_test for this disease
    if len(np.unique(y_test[:, i])) == 2:
        fpr, tpr, _ = roc_curve(y_test[:, i], y_proba[i][:, 1])
        roc_auc = roc_auc_score(y_test[:, i], y_proba[i][:, 1])
        plt.plot(fpr, tpr, label=f'{disease} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Disease')
plt.legend(loc='lower right')
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curves_multilabel.png'))
plt.close()

# Generate confusion matrices for each disease (optional)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()
for i, disease in enumerate(target_cols):
    cm = confusion_matrix(y_test[:, i], y_pred[:, i])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(disease)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrices_multilabel.png'))
plt.close()

print(f"Graphs saved to {RESULTS_DIR}")