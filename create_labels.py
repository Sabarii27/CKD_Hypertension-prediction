# create_labels.py
import os
import pandas as pd
from config import STARE_DIR

# Assume training images are disease (1) and validation are normal (0)
# You can change this if you have actual labels
split_csv = os.path.join(STARE_DIR, 'split_info.csv')
df_split = pd.read_csv(split_csv)

# Create label: 1 for training, 0 for validation
df_labels = df_split.copy()
df_labels['label'] = (df_split['split'] == 'training').astype(int)

# Save
csv_path = os.path.join(STARE_DIR, 'labels.csv')
df_labels.to_csv(csv_path, index=False)
print(f"Labels saved to {csv_path}")
print(df_labels)