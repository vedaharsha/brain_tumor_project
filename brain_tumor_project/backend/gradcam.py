import tensorflow as tf
import numpy as np
import cv2
import os

def generate_gradcam(model, img_array, original_image, output_dir):
    """
    img_array: (1, 224, 224, 3)
    original_image: numpy array (H,W,3)
    """

    last_conv_layer = None
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            last_conv_layer = layer.name
            break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

    filename = f"gradcam_{os.urandom(6).hex()}.jpg"
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, superimposed)

    return path
