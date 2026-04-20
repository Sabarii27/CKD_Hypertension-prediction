# extract_features_multilabel.py
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.filters import frangi
from skimage.morphology import skeletonize, remove_small_objects
from skimage import img_as_ubyte
from config import DATA_DIR, IMG_HEIGHT, IMG_WIDTH

def extract_vessel_features(img_path):
    """
    Load an image, preprocess, enhance vessels with Frangi, threshold,
    and compute morphological features.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    # If color, take green channel; otherwise use the image as is
    if len(img.shape) == 3:
        # Use green channel (index 1) for retinal images
        gray = img[:, :, 1]
    else:
        gray = img

    # Resize to standard size
    gray = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Frangi filter for vessel enhancement
    # Sigma range from 1 to 3 usually works for retinal vessels
    vesselness = frangi(enhanced, sigmas=range(1,4), scale_range=None, beta=0.5, gamma=15, black_ridges=False)

    # Normalize to 0-255 and threshold
    vesselness_norm = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-8)
    # Simple threshold (you may tune this later)
    binary = vesselness_norm > 0.2
    # Remove very small objects (noise)
    binary = remove_small_objects(binary, min_size=50)

    # --- Feature extraction from binary mask ---
    # Vessel density
    vessel_density = np.mean(binary)

    # Skeletonize to get vessel length
    skeleton = skeletonize(binary)
    vessel_length = np.sum(skeleton)

    # Branching points: pixels with >2 neighbors in the skeleton
    from scipy.signal import convolve2d
    kernel = np.ones((3,3))
    neighbor_count = convolve2d(skeleton.astype(int), kernel, mode='same', boundary='fill') - skeleton.astype(int)
    branching_points = np.sum((neighbor_count > 2) & skeleton)

    # Average vessel width (area / length)
    if vessel_length > 0:
        avg_width = np.sum(binary) / vessel_length
    else:
        avg_width = 0

    # Tortuosity (simplified: mean curvature of skeleton – placeholder)
    # We'll use a simple measure: ratio of skeleton length to Euclidean distance between endpoints
    # Not implemented here; we'll set a placeholder for now.
    tortuosity = 0.5  # placeholder

    # Additional intensity features from the preprocessed image
    mean_intensity = np.mean(enhanced)
    std_intensity = np.std(enhanced)

    return {
        'vessel_density': vessel_density,
        'vessel_length': vessel_length,
        'branching_points': branching_points,
        'avg_width': avg_width,
        'tortuosity': tortuosity,
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity
    }

def main():
    combined_dir = os.path.join(DATA_DIR, 'combined')
    img_dir = os.path.join(combined_dir, 'images')
    labels_csv = os.path.join(combined_dir, 'multilabel.csv')

    # Load image IDs from labels CSV
    df_labels = pd.read_csv(labels_csv)
    image_ids = df_labels['image_id'].tolist()

    print(f"Total images to process: {len(image_ids)}")

    features_list = []
    for img_id in tqdm(image_ids, desc="Extracting features"):
        # Find the image file (could be .jpg or .png)
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = os.path.join(img_dir, img_id + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            print(f"Warning: Image {img_id} not found, skipping.")
            continue

        feat = extract_vessel_features(img_path)
        if feat:
            feat['image_id'] = img_id
            features_list.append(feat)

    # Save features
    features_df = pd.DataFrame(features_list)
    out_path = os.path.join(combined_dir, 'features.csv')
    features_df.to_csv(out_path, index=False)
    print(f"Features saved to {out_path}")
    print(f"Processed {len(features_df)} images.")

if __name__ == '__main__':
    main()