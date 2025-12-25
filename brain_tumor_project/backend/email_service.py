import smtplib
import os
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

MAIL_USERNAME = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_SERVER = os.getenv("MAIL_SERVER")
MAIL_PORT = int(os.getenv("MAIL_PORT"))

def send_email_with_pdf(to_email, patient_name, pdf_path):
    msg = EmailMessage()
    msg["Subject"] = "AI Brain Tumor Report"
    msg["From"] = MAIL_USERNAME
    msg["To"] = to_email

    msg.set_content(
        f"""
Hello {patient_name},

Your AI-generated brain tumor report is attached.

Note: This report is AI-assisted and must be reviewed by a certified radiologist.

Regards,
AI Radiology System
"""
    )

    with open(pdf_path, "rb") as f:
        pdf_data = f.read()

    msg.add_attachment(
        pdf_data,
        maintype="application",
        subtype="pdf",
        filename="Brain_Tumor_Report.pdf"
    )

    # ✅ USE SMTP_SSL ONLY (NO TLS)
    with smtplib.SMTP_SSL(MAIL_SERVER, MAIL_PORT) as server:
        server.login(MAIL_USERNAME, MAIL_PASSWORD)
        server.send_message(msg)
