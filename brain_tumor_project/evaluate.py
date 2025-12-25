import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# --------------------------
# UPDATE DATASET PATHS ✔
# --------------------------
val_dir = r"C:\DL_PROJECT\brain_tumor_project\dataset\val"

# --------------------------
# Load model and labels
# --------------------------
model = load_model("models/best_model.h5")
print("Model Loaded!")

class_indices = np.load("models/class_indices.npy", allow_pickle=True).item()
labels = {v: k for k, v in class_indices.items()}  # reverse mapping

# --------------------------
# Predict function
# --------------------------
def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_arr = img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    prediction = model.predict(img_arr)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    tumor_type = labels[class_id]

    print(f"\nPrediction: {tumor_type.upper()}")
    print(f"Confidence: {confidence:.2f}%\n")

# --------------------------
# Main script
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input MRI image")
    args = parser.parse_args()

    predict_image(args.image)
