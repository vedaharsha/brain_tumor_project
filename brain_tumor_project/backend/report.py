from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from datetime import datetime
import os

def create_pdf(
    report_id,
    patient_name,
    patient_age,
    patient_email,
    tumor_type,
    confidence,
    probabilities,
    gradcam_path,
    ai_summary
):
    path = f"static/reports/{report_id}.pdf"
    c = canvas.Canvas(path, pagesize=A4)

    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 2 * cm, "AI Brain Tumor Report")

    # Patient Info
    c.setFont("Helvetica", 12)
    y = height - 4 * cm
    c.drawString(2 * cm, y, f"Patient Name: {patient_name}")
    c.drawString(2 * cm, y - 1 * cm, f"Age: {patient_age}")
    c.drawString(2 * cm, y - 2 * cm, f"Email: {patient_email}")
    c.drawString(2 * cm, y - 3 * cm, f"Generated On: {datetime.now()}")

    # Result
    y -= 5 * cm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2 * cm, y, "Diagnosis")

    c.setFont("Helvetica", 12)
    c.drawString(2 * cm, y - 1 * cm, f"Tumor Type: {tumor_type}")
    c.drawString(2 * cm, y - 2 * cm, f"Confidence: {confidence}%")

    # Probabilities
    y -= 4 * cm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2 * cm, y, "Class Probabilities")

    c.setFont("Helvetica", 12)
    offset = 1
    for k, v in probabilities.items():
        c.drawString(2 * cm, y - offset * cm, f"{k}: {v}%")
        offset += 1

    # AI Summary
    y -= (offset + 1) * cm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2 * cm, y, "AI Summary")

    c.setFont("Helvetica", 12)
    text = c.beginText(2 * cm, y - 1 * cm)
    for line in ai_summary.split(". "):
        text.textLine(line.strip())
    c.drawText(text)

    # GradCAM Image
    if os.path.exists(gradcam_path):
        c.showPage()
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(width / 2, height - 2 * cm, "Grad-CAM Visualization")
        c.drawImage(
            gradcam_path,
            2 * cm,
            height / 2 - 6 * cm,
            width - 4 * cm,
            10 * cm,
            preserveAspectRatio=True
        )

    c.save()
    return path
