# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import cv2
import os
import io
import matplotlib.pyplot as plt
from gtts import gTTS
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from datetime import datetime
import json
import base64

# -------------------- SETTINGS --------------------
MODEL_PATH = "models/best_model.h5"
LABELS_PATH = "models/class_indices.npy"
REPORTS_DIR = "reports"
IMG_SIZE = (224, 224)            # model input size
HEATMAP_ALPHA = 0.5

# Ensure reports folder exists
os.makedirs(REPORTS_DIR, exist_ok=True)

st.set_page_config(page_title="Brain Tumor Classification", layout="wide")

st.title("Brain Tumor Classification")
st.write("Upload one or more MRI images. The app will predict tumor type, show Grad-CAM heatmap and a bounding box, "
         "provide a downloadable report (PDF) and audio explanation.")

# -------------------- HELPER: load model --------------------
@st.cache_resource(show_spinner=False)
def load_my_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()
if model is None:
    st.warning("Model not loaded. Make sure models/best_model.h5 exists and is compatible with your TF/Keras version.")
else:
    st.success("Model loaded successfully ✔")

# Load class indices
try:
    class_indices = np.load(LABELS_PATH, allow_pickle=True).item()
    index_to_class = {v: k for k, v in class_indices.items()}
except Exception:
    index_to_class = None

# -------------------- GRAD-CAM --------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    """
    Returns a 2D heatmap numpy array resized to feature map size.
    If last_conv_layer_name is None: tries to pick the last conv layer.
    """
    if last_conv_layer_name is None:
        # try to find a conv layer automatically
        for layer in reversed(model.layers):
            if "conv" in layer.name or "Conv" in layer.__class__.__name__:
                last_conv_layer_name = layer.name
                break
    if last_conv_layer_name is None:
        raise ValueError("No convolutional layer found in model to compute Grad-CAM.")

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    else:
        heatmap = np.zeros_like(heatmap)
    return heatmap

def overlay_heatmap(original_image_bgr, heatmap, alpha=0.5):
    hmap_resized = cv2.resize(heatmap, (original_image_bgr.shape[1], original_image_bgr.shape[0]))
    hmap_uint8 = np.uint8(255 * hmap_resized)
    hmap_color = cv2.applyColorMap(hmap_uint8, cv2.COLORMAP_JET)
    # convert to same dtype
    overlay = cv2.addWeighted(hmap_color, alpha, original_image_bgr, 1 - alpha, 0)
    return overlay

def get_bounding_box_from_heatmap(heatmap, threshold=0.4):
    """Simple bounding box from heatmap by thresholding and finding largest contour"""
    hm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    hm_bin = (hm > threshold).astype(np.uint8) * 255
    # Resize to model input size if needed doesn't matter here; we will scale externally.
    contours, _ = cv2.findContours(hm_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # pick largest contour
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return (x, y, w, h), hm_bin

# -------------------- PDF report generation --------------------
def generate_pdf_report(orig_pil, heatmap_pil, chart_pil, prediction_text, confidence_text, auto_description, save_path):
    """
    Create a multi-page PDF containing original image, heatmap overlay, probability chart and description.
    No emojis in PDF text (we exclude them).
    """
    PAGE_W, PAGE_H = A4  # reportlab points (595x842)
    c = canvas.Canvas(save_path, pagesize=A4)

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(PAGE_W/2, PAGE_H - 40, "Radiology Department - AI Tumor Report")  # no emojis
    c.setFont("Helvetica", 10)
    c.drawString(40, PAGE_H - 60, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Left: original image; Right: heatmap
    margin = 40
    img_w = (PAGE_W - 3*margin) / 2
    img_h = img_w * (orig_pil.height / orig_pil.width)
    # ensure not too tall
    if img_h > 300:
        img_h = 300
        img_w = img_h * (orig_pil.width / orig_pil.height)

    # Draw original
    orig_reader = ImageReader(orig_pil)
    c.drawImage(orig_reader, margin, PAGE_H - 100 - img_h, width=img_w, height=img_h, preserveAspectRatio=True)

    # Draw heatmap
    heat_reader = ImageReader(heatmap_pil)
    c.drawImage(heat_reader, 2*margin + img_w, PAGE_H - 100 - img_h, width=img_w, height=img_h, preserveAspectRatio=True)

    # Prediction & confidence
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, PAGE_H - 120 - img_h, f"Prediction: {prediction_text}")
    c.setFont("Helvetica", 11)
    c.drawString(margin, PAGE_H - 140 - img_h, f"Confidence: {confidence_text}")

    # Draw the chart below
    chart_reader = ImageReader(chart_pil)
    chart_w = PAGE_W - 2*margin
    chart_h = 200
    c.drawImage(chart_reader, margin, PAGE_H - 160 - img_h - chart_h, width=chart_w, height=chart_h, preserveAspectRatio=True)

    # Auto-generated medical description
    text_top = PAGE_H - 180 - img_h - chart_h
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, text_top, "AI Summary / Impression:")
    c.setFont("Helvetica", 10)
    # wrap description
    text = c.beginText(margin, text_top - 18)
    text.setLeading(14)
    for line in auto_description.splitlines():
        text.textLine(line)
    c.drawText(text)

    c.showPage()
    c.save()
    return save_path

# -------------------- UTILS --------------------
def pil_to_bytes(pil_img, fmt="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    return buf

def save_bytesio_to_file(bio, path):
    with open(path, "wb") as f:
        f.write(bio.getbuffer())

# Small TTS via gTTS returning bytes
def generate_tts_bytes(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    bio = io.BytesIO()
    tts.write_to_fp(bio)
    bio.seek(0)
    return bio

# Probability chart using matplotlib (horizontal bars)
def make_probability_chart(predictions, labels):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, predictions * 100, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([lbl.upper() for lbl in labels])
    ax.invert_yaxis()
    ax.set_xlabel("Probability (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# Basic auto-generated medical description (simple template)
def generate_auto_description(pred_class, confidence, bbox=None):
    lines = []
    lines.append(f"The model predicts the presence of a {pred_class} with an estimated confidence of {confidence:.2f}%.")
    if bbox:
        lines.append("A focal region of increased activation was localized and a bounding box was computed to indicate the most suspicious area.")
    else:
        lines.append("No focal activation region could be confidently localized.")
    lines.append("Recommendation:")
    lines.append("- Correlate with clinical history and prior imaging if available.")
    lines.append("- Consider contrast-enhanced MRI or additional sequences if clinically indicated.")
    lines.append("- Multidisciplinary review is recommended for treatment planning.")
    return "\n".join(lines)

# -------------------- UI / Main --------------------
st.sidebar.header("Options")
use_last_conv = st.sidebar.text_input("Last conv layer name (optional)", value="")
heatmap_alpha = st.sidebar.slider("Heatmap alpha", 0.0, 1.0, float(HEATMAP_ALPHA), 0.05)

uploaded_files = st.file_uploader("Upload MRI images (JPG/PNG). You can select multiple files.", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploaded_files:
    # Drag-and-drop gallery: show thumbnails in a horizontal gallery
    st.markdown("### Gallery")
    cols = st.columns(min(6, len(uploaded_files)))
    for i, f in enumerate(uploaded_files):
        with cols[i % len(cols)]:
            img = Image.open(f).convert("RGB")
            st.image(img, width=150, caption=f.name)

    # Prepare space for prediction controls
    if st.button("Run Predictions and Generate Report"):
        # spinner & progress bar
        progress = st.progress(0)
        with st.spinner("Predicting... this may take a few seconds per image"):
            results = []
            total = len(uploaded_files)
            for idx, f in enumerate(uploaded_files):
                # update progress
                progress.progress(int((idx/total)*100))

                # read and preprocess
                pil_img = Image.open(f).convert("RGB")
                display_img = pil_img.copy()  # for show
                resized = pil_img.resize(IMG_SIZE)
                arr = img_to_array(resized) / 255.0
                arr_exp = np.expand_dims(arr, axis=0)

                # predict
                if model is None or index_to_class is None:
                    st.error("Model or labels not available. Prediction skipped.")
                    break
                preds = model.predict(arr_exp)[0]
                pred_idx = int(np.argmax(preds))
                pred_class = index_to_class.get(pred_idx, str(pred_idx))
                confidence = float(preds[pred_idx] * 100)

                # Grad-CAM
                try:
                    heatmap_small = make_gradcam_heatmap(arr_exp, model, last_conv_layer_name=use_last_conv or None)
                except Exception as e:
                    st.warning(f"Grad-CAM failed for {f.name}: {e}")
                    heatmap_small = np.zeros((IMG_SIZE[0]//16, IMG_SIZE[1]//16))  # fallback

                # Original as BGR for overlay
                orig_bgr = cv2.cvtColor(np.array(display_img), cv2.COLOR_RGB2BGR)
                overlay_bgr = overlay_heatmap(orig_bgr, heatmap_small, alpha=heatmap_alpha)
                overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
                overlay_pil = Image.fromarray(overlay_rgb)

                # bounding box from heatmap (on resized heatmap scaled to original image)
                # first scale heatmap_small to original image size for bbox detection
                hm_scaled = cv2.resize(heatmap_small, (display_img.width, display_img.height))
                bbox_info = get_bounding_box_from_heatmap(hm_scaled, threshold=0.4)
                bbox = None
                if bbox_info:
                    (x,y,w,h), _ = bbox_info
                    bbox = (x, y, w, h)
                    # draw rectangle on overlay_pil for visualization
                    overlay_draw = overlay_pil.copy()
                    draw_img = cv2.cvtColor(np.array(overlay_draw), cv2.COLOR_RGB2BGR)
                    cv2.rectangle(draw_img, (x,y), (x+w, y+h), (0,255,0), 3)
                    overlay_pil = Image.fromarray(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))

                # chart
                labels = [index_to_class[i] for i in range(len(preds))] if index_to_class else [str(i) for i in range(len(preds))]
                chart_img = make_probability_chart(preds, labels)

                # TTS (online)
                description_text = generate_auto_description(pred_class, confidence, bbox=bbox)
                try:
                    tts_bio = generate_tts_bytes(description_text)
                except Exception as e:
                    st.warning(f"Voice generation failed: {e}")
                    tts_bio = None

                # Save report assets locally (auto-save)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = f"{os.path.splitext(f.name)[0]}_{stamp}"
                # ensure safe filename
                safe_base = "".join([c if c.isalnum() or c in "._-" else "_" for c in base_name])
                img_orig_path = os.path.join(REPORTS_DIR, safe_base + "_orig.png")
                img_heat_path = os.path.join(REPORTS_DIR, safe_base + "_heat.png")
                chart_path = os.path.join(REPORTS_DIR, safe_base + "_chart.png")
                pdf_path = os.path.join(REPORTS_DIR, safe_base + "_report.pdf")
                audio_path = os.path.join(REPORTS_DIR, safe_base + "_explanation.mp3")

                display_img.save(img_orig_path)
                overlay_pil.save(img_heat_path)
                chart_img.save(chart_path)
                if tts_bio:
                    with open(audio_path, "wb") as af:
                        af.write(tts_bio.read())
                        tts_bio.seek(0)

                # Generate PDF
                try:
                    # Use images we just saved
                    pdf_path = generate_pdf_report(display_img, overlay_pil, chart_img,
                                                   prediction_text=pred_class,
                                                   confidence_text=f"{confidence:.2f}%",
                                                   auto_description=description_text,
                                                   save_path=pdf_path)
                except Exception as e:
                    st.warning(f"PDF generation failed: {e}")
                    pdf_path = None

                # metadata
                meta = {
                    "file_name": f.name,
                    "pred_class": pred_class,
                    "confidence": confidence,
                    "timestamp": stamp,
                    "pdf": pdf_path,
                    "orig": img_orig_path,
                    "heat": img_heat_path,
                    "chart": chart_path,
                    "audio": audio_path if tts_bio else None
                }
                # save JSON metadata for history
                meta_path = os.path.join(REPORTS_DIR, safe_base + "_meta.json")
                with open(meta_path, "w", encoding="utf-8") as jf:
                    json.dump(meta, jf, indent=2)

                results.append({
                    "meta": meta,
                    "pil_orig": display_img,
                    "pil_heat": overlay_pil,
                    "chart": chart_img,
                    "description": description_text,
                    "audio_bytes": tts_bio
                })

                # update progress
                progress.progress(int(((idx+1)/total)*100))

        # End spinner
        st.success("Predictions completed ✅")

        # Display results (side-by-side)
        for item in results:
            st.markdown("---")
            cols = st.columns([1,1,1])
            with cols[0]:
                st.markdown("**Uploaded Image**")
                st.image(item["pil_orig"], use_column_width=True, caption="Uploaded Image", clamp=True)
            with cols[1]:
                st.markdown("**Heatmap Overlay**")
                st.image(item["pil_heat"], use_column_width=True, caption="Heatmap Overlay", clamp=True)
            with cols[2]:
                st.markdown("**Probability Chart**")
                st.image(item["chart"], use_column_width=True, caption="Prediction Probabilities", clamp=True)

            # show text
            st.markdown("**Prediction Result**")
            st.write(f"**{item['meta']['pred_class'].upper()}**  —  Confidence: **{item['meta']['confidence']:.2f}%**")
            st.markdown("**AI summary:**")
            for L in item["description"].splitlines():
                st.write(L)

            # download buttons
            btn_cols = st.columns([1,1,1,1])
            # download heatmap
            heat_bytes = pil_to_bytes(item["pil_heat"], fmt="PNG")
            btn_cols[0].download_button(label="Download Heatmap",
                                       data=heat_bytes,
                                       file_name=os.path.basename(item["meta"]["heat"]),
                                       mime="image/png")
            # download original
            orig_bytes = pil_to_bytes(item["pil_orig"], fmt="PNG")
            btn_cols[1].download_button(label="Download Original",
                                       data=orig_bytes,
                                       file_name=os.path.basename(item["meta"]["orig"]),
                                       mime="image/png")
            # download chart
            chart_bytes = pil_to_bytes(item["chart"], fmt="PNG")
            btn_cols[2].download_button(label="Download Chart",
                                       data=chart_bytes,
                                       file_name=os.path.basename(item["meta"]["chart"]),
                                       mime="image/png")
            # download pdf
            if item["meta"]["pdf"] and os.path.exists(item["meta"]["pdf"]):
                with open(item["meta"]["pdf"], "rb") as pf:
                    pdf_data = pf.read()
                btn_cols[3].download_button(label="Download Report (PDF)",
                                           data=pdf_data,
                                           file_name=os.path.basename(item["meta"]["pdf"]),
                                           mime="application/pdf")

            # audio playback
            if item["audio_bytes"]:
                st.audio(item["audio_bytes"].read(), format="audio/mp3")

        # show reports folder quick link
        st.markdown(f"Saved reports and assets to `{REPORTS_DIR}/`")

# Show reports history (auto-saved)
st.sidebar.markdown("### Reports history")
if st.sidebar.button("Refresh history"):
    pass

# list saved reports metadata
meta_files = sorted([p for p in os.listdir(REPORTS_DIR) if p.endswith("_meta.json")], reverse=True)
if meta_files:
    for mf in meta_files[:10]:
        try:
            with open(os.path.join(REPORTS_DIR,mf), "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            st.sidebar.markdown(f"**{meta['file_name']}** — {meta['pred_class']} ({meta['confidence']:.2f}%)")
            link_pdf = meta.get("pdf")
            if link_pdf and os.path.exists(link_pdf):
                with open(link_pdf, "rb") as pf:
                    btn = st.sidebar.download_button(label=f"Download {os.path.basename(link_pdf)}", data=pf.read(),
                                                     file_name=os.path.basename(link_pdf), mime="application/pdf")
        except Exception:
            continue
else:
    st.sidebar.write("No saved reports found.")
