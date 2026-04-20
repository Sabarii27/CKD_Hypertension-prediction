# preprocess.py
import cv2
import numpy as np
import os
import csv
from tqdm import tqdm
from config import STARE_DIR, IMG_HEIGHT, IMG_WIDTH

def preprocess_image(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        return False
    green = img[:,:,1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(green)
    resized = cv2.resize(enhanced, (IMG_WIDTH, IMG_HEIGHT))
    normalized = resized / 255.0
    np.save(save_path, normalized)
    return True

def preprocess_mask(mask_path, save_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
    resized = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
    binary = (resized > 128).astype(np.float32)
    np.save(save_path, binary)
    return True

# Base directories
base_stare = os.path.join(STARE_DIR, 'STARE')
image_root = os.path.join(base_stare, 'images')
mask_root = os.path.join(base_stare, 'annotations')

# Output directories
preproc_img_dir = os.path.join(STARE_DIR, 'preprocessed')
preproc_mask_dir = os.path.join(STARE_DIR, 'masks_processed')
os.makedirs(preproc_img_dir, exist_ok=True)
os.makedirs(preproc_mask_dir, exist_ok=True)

# CSV to record which images belong to which split
csv_path = os.path.join(STARE_DIR, 'split_info.csv')

splits = ['training', 'validation']
success_count = 0
records = []

for split in splits:
    img_dir = os.path.join(image_root, split)
    mask_dir = os.path.join(mask_root, split)
    
    if not os.path.exists(img_dir):
        print(f"Warning: {img_dir} not found")
        continue
        
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.png')]
    print(f"Processing {split} set: {len(image_files)} images")
    
    for img_file in tqdm(image_files):
        base = os.path.splitext(img_file)[0]  # e.g., im0001
        img_path = os.path.join(img_dir, img_file)
        
        # Look for first observer mask: base + '.ah.png'
        mask_path = os.path.join(mask_dir, base + '.ah.png')
        if not os.path.exists(mask_path):
            # Fallback to second observer
            mask_path = os.path.join(mask_dir, base + '.vk.png')
        if not os.path.exists(mask_path):
            print(f"Mask not found for {img_file}, skipping.")
            continue
        
        img_save = os.path.join(preproc_img_dir, base + '.npy')
        mask_save = os.path.join(preproc_mask_dir, base + '.npy')
        
        if preprocess_image(img_path, img_save) and preprocess_mask(mask_path, mask_save):
            success_count += 1
            records.append([base, split])

print(f"Preprocessed {success_count} image-mask pairs.")

# Save split info
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_id', 'split'])
    writer.writerows(records)
print(f"Split info saved to {csv_path}")