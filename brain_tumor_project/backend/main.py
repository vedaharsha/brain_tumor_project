from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

import numpy as np
import cv2
from PIL import Image
import io
import uuid
import os

from model import predict_tumor
from gradcam import generate_gradcam
from report import create_pdf
from email_service import send_email_with_pdf

# ---------------- APP SETUP ---------------- #

app = FastAPI(title="Brain Tumor Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure folders exist
os.makedirs("static/gradcam", exist_ok=True)
os.makedirs("static/reports", exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------- AI SUMMARY ---------------- #

AI_SUMMARY = {
    "glioma": "Glioma detected. This tumor originates from glial cells and may require further MRI and biopsy for grading.",
    "meningioma": "Meningioma detected. Often slow-growing and usually benign, but clinical evaluation is recommended.",
    "pituitary": "Pituitary tumor detected. Hormonal evaluation and endocrinology consultation are advised.",
    "no_tumor": "No tumor detected. MRI appears normal, but clinical correlation is recommended."
}

# ---------------- PREDICT ENDPOINT ---------------- #

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    patient_name: str = Form(...),
    patient_age: str = Form(...),
    patient_email: str = Form(...)
):
    try:
        print("Predict API called")

        # ---------- READ IMAGE (ONLY ONCE) ----------
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(pil_image)

        # ---------- PREPROCESS ----------
        resized = cv2.resize(img_array, (224, 224))
        normalized = resized / 255.0
        model_input = np.expand_dims(normalized, axis=0)

        # ---------- PREDICTION ----------
        prediction, confidence, probabilities, model = predict_tumor(model_input)

        # ---------- GRADCAM ----------
        gradcam_path = generate_gradcam(
            model=model,
            img_array=model_input,
            original_image=img_array,
            output_dir="static/gradcam"
        )

        gradcam_url = f"/static/gradcam/{os.path.basename(gradcam_path)}"

        # ---------- AI SUMMARY ----------
        ai_summary = AI_SUMMARY.get(prediction, "Clinical correlation required.")

        # ---------- PDF REPORT ----------
        report_id = str(uuid.uuid4())
        pdf_path = create_pdf(
            report_id=report_id,
            patient_name=patient_name,
            patient_age=patient_age,
            patient_email=patient_email,
            tumor_type=prediction,
            confidence=confidence,
            probabilities=probabilities,
            gradcam_path=gradcam_path,
            ai_summary=ai_summary
        )

        pdf_url = f"/static/reports/{os.path.basename(pdf_path)}"

        # ---------- EMAIL ----------
        send_email_with_pdf(
            to_email=patient_email,
            patient_name=patient_name,
            pdf_path=pdf_path
        )

        # ---------- RESPONSE ----------
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "ai_summary": ai_summary,
            "gradcam_url": gradcam_url,
            "pdf_url": pdf_url
        }

    except Exception as e:
        print("ERROR:", str(e))
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
