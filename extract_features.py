# extract_features.py
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from skimage.morphology import skeletonize, binary_dilation, disk
from scipy import ndimage
from config import STARE_DIR

def extract_features_from_mask(mask):
    """Compute vessel features from binary mask (0=background, 1=vessel)."""
    # Vessel density
    vessel_density = np.mean(mask)
    
    # Skeletonize to get vessel length
    binary = mask > 0.5
    skeleton = skeletonize(binary)
    vessel_length = np.sum(skeleton)
    
    # Branching points (pixels with >2 neighbors in skeleton)
    # Use convolution to count neighbors
    from scipy.signal import convolve2d
    kernel = np.ones((3,3))
    neighbor_count = convolve2d(skeleton.astype(int), kernel, mode='same', boundary='fill') - skeleton.astype(int)
    branching = np.sum((neighbor_count > 2) & skeleton)
    
    # Average vessel width (area / length)
    if vessel_length > 0:
        avg_width = np.sum(binary) / vessel_length
    else:
        avg_width = 0
    
    # Tortuosity: simplified as curvature of skeleton
    # Not implemented in simple version; we'll use a placeholder
    tortuosity = 0.5  # placeholder
    
    return {
        'vessel_density': vessel_density,
        'vessel_length': vessel_length,
        'branching_points': branching,
        'avg_width': avg_width,
        'tortuosity': tortuosity
    }

# Load segmentation masks (ground truth from preprocessing) OR use predicted masks?
# For simplicity, we'll use the ground truth masks we already have in masks_processed.
# This ensures we have clean features, but in a real system we'd use predicted masks.
mask_dir = os.path.join(STARE_DIR, 'masks_processed')
image_ids = [f.replace('.npy', '') for f in os.listdir(mask_dir) if f.endswith('.npy')]

features_list = []
for img_id in tqdm(image_ids):
    mask_path = os.path.join(mask_dir, img_id + '.npy')
    mask = np.load(mask_path)
    feat = extract_features_from_mask(mask)
    feat['image_id'] = img_id
    features_list.append(feat)

df = pd.DataFrame(features_list)
df.to_csv(os.path.join(STARE_DIR, 'features.csv'), index=False)
print("Features saved to", os.path.join(STARE_DIR, 'features.csv'))