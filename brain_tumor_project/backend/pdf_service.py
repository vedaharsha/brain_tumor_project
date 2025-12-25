from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from datetime import datetime
import os

def create_pdf_report(
    pdf_path,
    patient,
    tumor_type,
    confidence,
    original_img,
    gradcam_img,
    ai_summary
):
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height-40, "AI Radiology Tumor Report")

    c.setFont("Helvetica", 11)
    c.drawString(40, height-80, f"Patient Name: {patient['name']}")
    c.drawString(40, height-100, f"Age: {patient['age']}")
    c.drawString(40, height-120, f"Email: {patient['email']}")
    c.drawString(40, height-140, f"Generated On: {datetime.now()}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, height-180, f"Tumor Type: {tumor_type.upper()}")
    c.drawString(40, height-200, f"Confidence: {confidence:.2f}%")

    # Images
    c.drawImage(ImageReader(original_img), 40, height-460, 230, 230, preserveAspectRatio=True)
    c.drawImage(ImageReader(gradcam_img), 300, height-460, 230, 230, preserveAspectRatio=True)

    # AI Summary
    text = c.beginText(40, height-520)
    text.setFont("Helvetica", 11)
    for line in ai_summary.split("\n"):
        text.textLine(line)
    c.drawText(text)

    c.save()
