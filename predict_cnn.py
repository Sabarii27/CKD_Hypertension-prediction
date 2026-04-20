# predict_cnn.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os
import sys

# Paths – adjust if needed
MODEL_PATH = os.path.join('models', 'end_to_end_cnn.h5')
IMG_SIZE = 224
target_columns = ['ckd', 'hypertension', 'diabetes', 'amd', 'glaucoma', 'cataract', 'myopia', 'normal']

def load_and_preprocess_image(image_path):
    """Load an image, resize, normalize, and convert to RGB if needed."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    # Convert BGR to RGB (OpenCV loads as BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    # Normalize to [0,1]
    img_norm = img_resized.astype(np.float32) / 255.0
    # Add batch dimension
    img_batch = np.expand_dims(img_norm, axis=0)
    return img_batch

def predict_image(image_path):
    """Load model and predict diseases for a single image."""
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Please check the path.")
        return
    model = load_model(MODEL_PATH)
    print("Model loaded.")
    
    try:
        img = load_and_preprocess_image(image_path)
    except Exception as e:
        print(e)
        return
    
    # Predict – model returns a list of 8 arrays, each shape (1,1)
    preds = model.predict(img, verbose=0)
    # Convert to probabilities (already sigmoid output)
    probs = np.array([p[0][0] for p in preds])  # shape (8,)
    
    print("\n=== Prediction Results ===")
    for name, prob in zip(target_columns, probs):
        status = "POSITIVE" if prob >= 0.5 else "NEGATIVE"
        print(f"{name:12s}: {status}  (confidence: {prob:.4f})")
    
    # Show the image (optional, requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        img_disp = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        plt.imshow(img_disp)
        plt.title(f"CKD: {probs[0]:.2f}, HTN: {probs[1]:.2f}")
        plt.axis('off')
        plt.show()
    except ImportError:
        pass

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter path to retinal image: ").strip()
    predict_image(image_path)