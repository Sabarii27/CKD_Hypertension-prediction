# gradcam.py
import numpy as np
import tensorflow as tf

def saliency_map(img_array, model, disease_index):
    """
    Compute a saliency map for the specified disease index.
    Highlights pixels with the highest gradient impact.
    """
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        # Forward pass – model returns a list of 8 tensors, each shape (1,1)
        outputs = model(img_tensor, training=False)
        # Extract the scalar prediction for the chosen disease
        loss = outputs[disease_index][0, 0]

    grads = tape.gradient(loss, img_tensor)
    # Take maximum absolute gradient across colour channels
    saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()[0]
    # Normalize to [0,1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency