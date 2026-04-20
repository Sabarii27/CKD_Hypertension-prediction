# train_segmentation.py
import numpy as np
import os
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.model_selection import train_test_split
from unet_model import unet
from config import STARE_DIR, MODELS_DIR, RESULTS_DIR, IMG_HEIGHT, IMG_WIDTH

# Load preprocessed data
preproc_img_dir = os.path.join(STARE_DIR, 'preprocessed')
preproc_mask_dir = os.path.join(STARE_DIR, 'masks_processed')
split_csv = os.path.join(STARE_DIR, 'split_info.csv')

# Get list of image IDs
df_split = pd.read_csv(split_csv)
image_ids = df_split['image_id'].tolist()

print(f"Image IDs: {image_ids}")

X = []
y = []
for img_id in image_ids:
    img_path = os.path.join(preproc_img_dir, img_id + '.npy')
    mask_path = os.path.join(preproc_mask_dir, img_id + '.npy')
    if os.path.exists(img_path) and os.path.exists(mask_path):
        img_data = np.load(img_path)
        mask_data = np.load(mask_path)
        print(f"Loaded {img_id}: image shape {img_data.shape}, mask shape {mask_data.shape}")
        X.append(img_data)
        y.append(mask_data)
    else:
        print(f"Missing: {img_path} or {mask_path}")

if len(X) == 0:
    raise ValueError("No data loaded. Check preprocessing.")

X = np.array(X)
y = np.array(y)

print(f"X shape before reshape: {X.shape}, y shape: {y.shape}")

X = X.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
y = y.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)

print(f"X shape after reshape: {X.shape}, y shape: {y.shape}")

# Split into train/val using the predefined split
train_ids = df_split[df_split['split'] == 'training']['image_id'].tolist()
val_ids = df_split[df_split['split'] == 'validation']['image_id'].tolist()

train_idx = [i for i, img_id in enumerate(image_ids) if img_id in train_ids]
val_idx = [i for i, img_id in enumerate(image_ids) if img_id in val_ids]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

print(f"Train indices: {train_idx}, Val indices: {val_idx}")
print(f"X_train type: {type(X_train)}, shape: {X_train.shape}")
print(f"X_val type: {type(X_val)}, shape: {X_val.shape}")

model = unet()
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint(os.path.join(MODELS_DIR, 'segmentation_best.h5'), save_best_only=True),
    EarlyStopping(patience=20, restore_best_weights=True),
    CSVLogger(os.path.join(RESULTS_DIR, 'training_log.csv'))
]

print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=2,
    callbacks=callbacks,
    verbose=1
)

model.save(os.path.join(MODELS_DIR, 'segmentation_final.h5'))
print("Training complete. Models saved.")