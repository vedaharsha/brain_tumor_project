import tensorflow as tf
import numpy as np
import cv2
import os
import uuid

MODEL_LAYER_NAME = "conv5_block3_out"  # change if your model differs
STATIC_DIR = "static/gradcam"

os.makedirs(STATIC_DIR, exist_ok=True)

def generate_gradcam(img_array, model):
    """
    img_array: numpy array of shape (1, 224, 224, 3)
    model: loaded keras model
    """

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(MODEL_LAYER_NAME).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

    # Convert to image
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = np.uint8(img_array[0] * 255)
    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    filename = f"{uuid.uuid4()}.jpg"
    path = os.path.join(STATIC_DIR, filename)
    cv2.imwrite(path, superimposed)

    return path
