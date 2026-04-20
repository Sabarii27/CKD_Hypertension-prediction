# predict_multilabel.py
import cv2
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from skimage.filters import frangi
from skimage.morphology import skeletonize, remove_small_objects
from scipy.signal import convolve2d
from config import MODELS_DIR, STARE_DIR, IMG_HEIGHT, IMG_WIDTH

def preprocess_and_segment(img_path, seg_model):
    """
    Load image, extract green channel, enhance, and run segmentation model.
    Returns enhanced grayscale image and predicted vessel mask.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    # Green channel
    green = img[:,:,1]
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(green)
    # Resize
    resized = cv2.resize(enhanced, (IMG_WIDTH, IMG_HEIGHT))
    # Normalize for segmentation model
    norm = resized / 255.0
    # Predict vessel mask
    pred = seg_model.predict(norm.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1), verbose=0)[0,:,:,0]
    
    # Debug: print mask statistics
    print(f"Vessel mask shape: {pred.shape}, min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}")
    
    return resized, pred

def extract_features_from_mask(img_gray):
    """
    Use Frangi filter to enhance vessels, then compute morphological features.
    Returns a feature vector matching the one used during training.
    """
    # Frangi filter for vessel enhancement
    vesselness = frangi(img_gray, sigmas=range(1,4), beta=0.5, gamma=15, black_ridges=False)
    vesselness_norm = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-8)
    binary = vesselness_norm > 0.2
    binary = remove_small_objects(binary, min_size=50)

    # Vessel density
    vessel_density = np.mean(binary)

    # Skeletonize
    skeleton = skeletonize(binary)
    vessel_length = np.sum(skeleton)

    # Branching points
    kernel = np.ones((3,3))
    neighbor_count = convolve2d(skeleton.astype(int), kernel, mode='same') - skeleton.astype(int)
    branching_points = np.sum((neighbor_count > 2) & skeleton)

    # Average width
    if vessel_length > 0:
        avg_width = np.sum(binary) / vessel_length
    else:
        avg_width = 0

    # Tortuosity (placeholder)
    tortuosity = 0.5

    # Intensity statistics
    mean_intensity = np.mean(img_gray)
    std_intensity = np.std(img_gray)

    # Return as a 2D array (1 sample, 7 features)
    return np.array([[vessel_density, vessel_length, branching_points, avg_width,
                      tortuosity, mean_intensity, std_intensity]])

def main():
    # Load models
    print("Loading segmentation model...")
    seg_model = load_model(os.path.join(MODELS_DIR, 'segmentation_best.h5'))
    print("Loading classifier...")
    clf = joblib.load(os.path.join(MODELS_DIR, 'multilabel_classifier.pkl'))

    # Get image path from user or use a default
    img_path = input("Enter path to retinal image (or press Enter for default test image): ").strip()
    if not img_path:
        # Use a validation image from STARE if available
        default_img = os.path.join(STARE_DIR, 'STARE', 'images', 'validation', 'im0162.png')
        if os.path.exists(default_img):
            img_path = default_img
        else:
            print("No default image found. Please provide a valid path.")
            return

    if not os.path.exists(img_path):
        print("File not found. Exiting.")
        return

    print(f"Processing image: {img_path}")

    # Preprocess and segment
    gray, vessel_mask = preprocess_and_segment(img_path, seg_model)
    if gray is None:
        print("Failed to load image.")
        return

    # Extract features (using Frangi – same as training)
    features = extract_features_from_mask(gray)

    # Get prediction probabilities (list of arrays, one per output)
    pred_proba = clf.predict_proba(features)  # list of 8 arrays, each shape (1,2)
    pred_labels = clf.predict(features)[0]    # array of 8 binary predictions

    # Indices for CKD and hypertension (based on training order)
    # Order: ckd, hypertension, diabetes, amd, glaucoma, cataract, myopia, normal
    disease_names = ['CKD', 'Hypertension']
    indices = [0, 1]

    print("\n==========================================")
    print("     CKD & HYPERTENSION PREDICTION")
    print("==========================================")
    for i, name in zip(indices, disease_names):
        prob = pred_proba[i][0][1]   # probability of positive class
        status = "POSITIVE" if pred_labels[i] == 1 else "NEGATIVE"
        print(f"{name:12s}: {status}  (confidence: {prob:.2f})")
    print("==========================================")

    # Create a figure with original, vessel mask, and predictions
    original = cv2.imread(img_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # If vessel_mask is all zeros or very faint, try to use Frangi for display
    if vessel_mask.mean() < 0.01:
        print("Segmentation model gave a weak mask; using Frangi for display.")
        # Compute Frangi on the grayscale image
        vesselness = frangi(gray, sigmas=range(1,4), beta=0.5, gamma=15, black_ridges=False)
        vesselness_norm = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-8)
        display_mask = (vesselness_norm > 0.2).astype(np.float32)
    else:
        display_mask = vessel_mask

    plt.figure(figsize=(12, 4))

    plt.subplot(1,3,1)
    plt.imshow(original_rgb)
    plt.title('Original Retinal Image')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(display_mask, cmap='gray')
    plt.title('Vessel Segmentation')
    plt.axis('off')

    plt.subplot(1,3,3)
    # Create text summary for CKD and hypertension only
    text_lines = []
    for i, name in zip(indices, disease_names):
        status = "POSITIVE" if pred_labels[i] == 1 else "NEGATIVE"
        prob = pred_proba[i][0][1]
        text_lines.append(f"{name}: {status}\nConfidence: {prob:.2f}")
    text = "\n\n".join(text_lines)
    plt.text(0.1, 0.5, text, fontsize=12, transform=plt.gca().transAxes,
             verticalalignment='center', family='monospace', weight='bold')
    plt.title('Prediction Results')
    plt.axis('off')

    plt.tight_layout()
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    out_path = os.path.join('results', 'ckd_htn_prediction.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nOutput image saved to {out_path}")
    plt.show()

    # Optionally save the vessel mask separately for inspection
    cv2.imwrite(os.path.join('results', 'vessel_mask.png'), (display_mask * 255).astype(np.uint8))
    print("Vessel mask also saved to results/vessel_mask.png")

if __name__ == '__main__':
    main()