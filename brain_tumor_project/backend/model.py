import tensorflow as tf
import numpy as np

MODEL_PATH = "models/best_model.h5"

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]

# Load model ONCE
model = tf.keras.models.load_model(MODEL_PATH)

def predict_tumor(img_array):
    """
    img_array shape: (1, 224, 224, 3)
    """
    preds = model.predict(img_array)[0]

    class_index = int(np.argmax(preds))
    prediction = CLASS_NAMES[class_index]
    confidence = round(float(preds[class_index] * 100), 2)

    probabilities = {
        CLASS_NAMES[i]: round(float(preds[i] * 100), 2)
        for i in range(len(CLASS_NAMES))
    }

    return prediction, confidence, probabilities, model
