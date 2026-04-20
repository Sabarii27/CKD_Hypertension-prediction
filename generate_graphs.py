# generate_graphs.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import joblib
from config import STARE_DIR, MODELS_DIR, RESULTS_DIR

# Load classifier and data
clf = joblib.load(os.path.join(MODELS_DIR, 'classifier.pkl'))
features = pd.read_csv(os.path.join(STARE_DIR, 'features.csv'))
labels = pd.read_csv(os.path.join(STARE_DIR, 'labels.csv'))
df = features.merge(labels, on='image_id', how='inner')
feature_cols = [col for col in df.columns if col not in ['image_id', 'split', 'label']]
X = df[feature_cols].values
y = df['label'].values
y_prob = clf.predict_proba(X)[:,1]
y_pred = clf.predict(X)

# Confusion matrix on whole dataset (since small)
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal','Disease'], yticklabels=['Normal','Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'))
plt.close()

print("Graphs saved to", RESULTS_DIR)