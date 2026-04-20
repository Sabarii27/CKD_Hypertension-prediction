# train_classifier.py
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
from config import STARE_DIR, MODELS_DIR, RESULTS_DIR

# Load features and labels
features = pd.read_csv(os.path.join(STARE_DIR, 'features.csv'))
labels = pd.read_csv(os.path.join(STARE_DIR, 'labels.csv'))

# Merge on image_id
df = features.merge(labels, on='image_id', how='inner')
print(f"Total samples: {len(df)}")
print("Columns:", df.columns.tolist())

# Keep only numeric feature columns (exclude image_id, split, and any other non-numeric)
feature_cols = [col for col in df.columns if col not in ['image_id', 'split', 'label']]
X = df[feature_cols].values
y = df['label'].values

print(f"Feature matrix shape: {X.shape}")

# Split (use same train/val split as before, or random)
# We'll use random split since we have split column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:,1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal','Disease']))

# Save model
joblib.dump(clf, os.path.join(MODELS_DIR, 'classifier.pkl'))

# Save results
with open(os.path.join(RESULTS_DIR, 'classification_results.txt'), 'w') as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"AUC: {auc:.4f}\n")
    f.write(f"Confusion Matrix:\n{cm}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=['Normal','Disease']))