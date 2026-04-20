# app.py – Clean layout: logo + nav inline, dark form containers, sticky footer
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
from datetime import datetime
from fpdf import FPDF
import tempfile
import os
from gradcam import saliency_map

# ---------- Page Configuration ----------
st.set_page_config(page_title="CKD & Hypertension Predictor", layout="wide", initial_sidebar_state="collapsed")

# ---------- Custom CSS ----------
st.markdown("""
<style>
    /* Gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    }
    
    /* Hide default sidebar */
    [data-testid="collapsedControl"] {
        display: none;
    }
    
    /* Remove extra top padding from the main container */
    [data-testid="stAppViewContainer"] {
        padding-top: 0 !important;
    }
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Main content area */
    .main-content {
        padding: 10px 40px 80px 40px; /* reduced top padding */
        color: #f1f5f9;
    }
    
    /* Headings and text */
    .main-content h1, .main-content h2, .main-content h3 {
        color: white;
        margin-top: 0;
    }
    .main-content p, .main-content li, .main-content .stMarkdown {
        color: #e2e8f0;
        font-size: 1.05rem;
        line-height: 1.5;
    }
    
    /* Dark container for prediction form and results */
    .dark-container {
        background: rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(8px);
        border-radius: 20px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Form inputs – dark theme */
    .stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox div {
        background-color: #1e293b !important;
        color: white !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }
    .stTextInput label, .stTextArea label, .stNumberInput label, .stSelectbox label {
        color: #cbd5e1 !important;
    }
    
    /* Buttons */
    .stButton button {
        background: #f97316;
        color: white;
        border: none;
        border-radius: 30px;
        padding: 6px 20px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton button:hover {
        background: #fdba74;
        color: #0f172a;
    }
    
    /* Navigation buttons (inline) */
    .nav-buttons {
        display: flex;
        gap: 15px;
        justify-content: flex-end;
        margin-bottom: 30px;
    }
    .nav-buttons button {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(4px);
        border: none;
        border-radius: 40px;
        padding: 6px 20px;
        color: white;
        font-weight: 500;
    }
    .nav-buttons button:hover {
        background: #f97316;
    }
    
    /* Image styling */
    .stImage img {
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    
    /* Sticky footer */
    .footer {
        text-align: center;
    margin-top: 40px;
    padding: 15px;
    font-size: 0.8rem;
    color: #94a3b8;
    background: rgba(0,0,0,0.5);
    border-radius: 12px;
    border-top: 1px solid #334155;
    }
    
    /* Ensure main content doesn't overlap footer */
    .main-content {
        padding-bottom: 80px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Load Model (cached) ----------
MODEL_PATH = 'models/end_to_end_cnn.h5'
IMG_SIZE = 224
full_target_columns = ['ckd', 'hypertension', 'diabetes', 'amd', 'glaucoma', 'cataract', 'myopia', 'normal']
display_diseases = ['ckd', 'hypertension', 'normal']

@st.cache_resource
def load_model_cached():
    model = load_model(MODEL_PATH)
    dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    model(dummy)
    return model

model = load_model_cached()

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

def get_predictions(img_array):
    preds = model.predict(img_array, verbose=0)
    probs = np.array([p[0][0] for p in preds])
    return probs

def clean_text(txt):
    return txt.replace('\u2013', '-').replace('\u2014', '-').replace('\u2018', "'").replace('\u2019', "'")

def generate_pdf(user_data, probs, image, img_array, model):
    pdf = FPDF(format='A4', unit='mm')
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Retinal Disease Prediction Report", ln=1, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', size=11)
    pdf.cell(200, 10, txt="User Details", ln=1)
    pdf.set_font("Arial", size=10)
    for key, value in user_data.items():
        pdf.cell(200, 8, txt=clean_text(f"{key}: {value}"), ln=1)
    pdf.ln(5)

    idx_ckd = full_target_columns.index('ckd')
    idx_htn = full_target_columns.index('hypertension')
    idx_norm = full_target_columns.index('normal')
    raw_ckd = probs[idx_ckd]
    raw_htn = probs[idx_htn]
    raw_norm = probs[idx_norm]
    exp_values = np.exp([raw_ckd, raw_htn, raw_norm])
    percentages = exp_values / exp_values.sum() * 100
    ckd_percent = percentages[0]
    htn_percent = percentages[1]
    norm_percent = percentages[2]

    pdf.set_font("Arial", 'B', size=11)
    pdf.cell(200, 10, txt="Prediction Results (CKD & Hypertension)", ln=1)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 8, txt=clean_text(f"CKD Probability: {ckd_percent:.1f}%"), ln=1)
    pdf.cell(200, 8, txt=clean_text(f"Hypertension Probability: {htn_percent:.1f}%"), ln=1)
    pdf.cell(200, 8, txt=clean_text(f"Normal Probability: {norm_percent:.1f}%"), ln=1)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', size=11)
    pdf.cell(200, 10, txt="Recommendation", ln=1)
    pdf.set_font("Arial", size=10)

    if ckd_percent > htn_percent:
        pdf.cell(200, 8, txt=clean_text("Based on your retinal analysis, you have a higher risk for Chronic Kidney Disease (CKD). We recommend:"), ln=1)
        recs = [
            "1. Consult a nephrologist for a comprehensive kidney function evaluation (blood tests, urine analysis).",
            "2. Monitor your blood pressure regularly - uncontrolled hypertension accelerates CKD progression.",
            "3. Maintain a kidney-friendly diet: reduce salt, processed foods, and protein intake as advised.",
            "4. Stay hydrated and avoid NSAIDs (painkillers like ibuprofen) which can harm kidneys.",
            "5. Schedule regular follow-ups every 6-12 months to track eGFR and albuminuria levels."
        ]
    elif htn_percent > ckd_percent:
        pdf.cell(200, 8, txt=clean_text("Based on your retinal analysis, you have a higher risk for Hypertension. We recommend:"), ln=1)
        recs = [
            "1. Measure your blood pressure at home and keep a log - target <130/80 mmHg.",
            "2. Reduce sodium intake (less than 2,300 mg/day) and adopt the DASH diet (fruits, vegetables, low-fat dairy).",
            "3. Engage in regular physical activity (30 minutes daily, 5 days/week).",
            "4. Limit alcohol and quit smoking - both raise blood pressure.",
            "5. Consult a physician for possible medication and routine blood pressure monitoring every 3-6 months."
        ]
    else:
        pdf.cell(200, 8, txt=clean_text("Your risk for CKD and Hypertension is similar. We recommend:"), ln=1)
        recs = [
            "1. Get a full health check-up including blood pressure and kidney function tests.",
            "2. Adopt a balanced, low-salt diet and maintain a healthy weight.",
            "3. Exercise regularly (at least 150 minutes of moderate activity per week).",
            "4. Avoid excessive alcohol and smoking.",
            "5. Re-evaluate annually or earlier if symptoms appear (fatigue, swelling, headaches)."
        ]
    for rec in recs:
        pdf.cell(200, 8, txt=clean_text(rec), ln=1)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', size=11)
    pdf.cell(200, 10, txt="Image Analysis", ln=1)
    pdf.ln(2)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_orig:
        image.save(tmp_orig.name)
        pdf.image(tmp_orig.name, x=50, w=100)
        orig_path = tmp_orig.name

    if ckd_percent != htn_percent:
        higher = "CKD" if ckd_percent > htn_percent else "Hypertension"
        disease_idx = full_target_columns.index(higher.lower())
        heatmap = saliency_map(img_array, model, disease_idx)
        heatmap_resized = cv2.resize(heatmap, (image.width, image.height))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        superimposed = np.array(image) * 0.6 + heatmap_color * 0.4
        superimposed = superimposed.astype(np.uint8)
        heatmap_img = Image.fromarray(superimposed)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_heat:
            heatmap_img.save(tmp_heat.name)
            pdf.ln(10)
            pdf.cell(200, 10, txt=clean_text(f"Heatmap for {higher} (highlighted regions)"), ln=1)
            pdf.image(tmp_heat.name, x=50, w=100)
            heatmap_path = tmp_heat.name

    pdf.output("report.pdf")
    os.unlink(orig_path)
    if ckd_percent != htn_percent:
        os.unlink(heatmap_path)
    return "report.pdf"

# ---------- Main Content ----------
st.markdown("<div class='main-content'>", unsafe_allow_html=True)

# Logo and navigation (inline, no extra box)
col_logo, col_nav = st.columns([1, 3])
with col_logo:
    st.markdown("<div style='display: flex; align-items: center; gap: 10px;'><div style='font-size: 2.5rem; background: white; border-radius: 50%; width: 55px; height: 55px; text-align: center; line-height: 55px;'>🩺</div><div style='font-size: 1.5rem; font-weight: bold; color: white;'>RetinaRisk AI</div></div>", unsafe_allow_html=True)
with col_nav:
    nav_cols = st.columns(3)
    with nav_cols[0]:
        if st.button("🏠 Home", key="nav_home", use_container_width=True):
            st.session_state.page = "Home"
    with nav_cols[1]:
        if st.button("ℹ️ About", key="nav_about", use_container_width=True):
            st.session_state.page = "About"
    with nav_cols[2]:
        if st.button("🔍 Prediction", key="nav_pred", use_container_width=True):
            st.session_state.page = "Prediction"

# Set default page if not set
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---------- Page Content ----------
if st.session_state.page == "Home":
    st.title("🩺 Retinal Disease Predictor")
    st.markdown("""
    ### Welcome to the CKD & Hypertension Prediction System
    This AI‑powered tool analyses retinal fundus images to predict the risk of:
    - **Chronic Kidney Disease (CKD)**
    - **Hypertension**

    **How it works:**
    1. Enter your personal details (name, age, etc.)
    2. Upload a retinal fundus image
    3. Get instant predictions with explainable heatmaps (saliency maps)
    4. Download a PDF report for your records

    This system is designed for non‑invasive, accessible screening – especially useful in remote or resource‑limited settings.
    """)
    st.image("example_retina.jpg", caption="Example retinal image", width=400)
    

elif st.session_state.page == "About":
    st.title("About the Project")
    st.markdown("""
    **Hypertension and Chronic Kidney Disease Prediction using Retinal Vessel Segmentation and Deep Learning**

    **Team:** [Your Names]  
    **Guide:** [Guide Name]

    **Abstract:**  
    This project leverages retinal vessel analysis and deep learning to predict hypertension and CKD non‑invasively. Using a ResNet50‑based multi‑label classifier trained on over 16,000 fundus images, the model achieves 94.6% accuracy for CKD and 99.9% for hypertension. The system includes a Streamlit web app with saliency maps for explainability, and generates downloadable patient reports.

    **Technology Stack:**
    - Python, TensorFlow/Keras, Streamlit
    - OpenCV, PIL, FPDF
    - Google Colab (training) / VS Code (deployment)

    **Contact:**  
    For queries, please reach out to [your email].
    """)

elif st.session_state.page == "Prediction":
    st.title("🔍 Prediction")
    st.markdown("Please fill in your details and upload a retinal image.")

    # Dark container for the whole prediction form
    st.markdown("<div class='dark-container'>", unsafe_allow_html=True)

    with st.form("user_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name")
            age = st.number_input("Age", min_value=1, max_value=120, step=1)
            email = st.text_input("Email")
        with col2:
            address = st.text_area("Address")
            phone = st.text_input("Phone Number")
        submitted = st.form_submit_button("Submit & Proceed")

    if submitted:
        if not name or not email or not phone:
            st.error("Please fill all required fields (Name, Email, Phone).")
        else:
            st.session_state.user_data = {
                "Name": name,
                "Age": age,
                "Address": address,
                "Email": email,
                "Phone": phone
            }
            st.session_state.user_form_submitted = True
            st.success("Details saved. Now upload an image.")

    if st.session_state.get("user_form_submitted", False):
        st.subheader("Upload Retinal Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

        if uploaded_file:
            img_array, original_img = preprocess_image(uploaded_file)
            st.image(original_img, caption='Uploaded Image', width=400)

            probs = get_predictions(img_array)

            idx_ckd = full_target_columns.index('ckd')
            idx_htn = full_target_columns.index('hypertension')
            idx_norm = full_target_columns.index('normal')
            raw_ckd = probs[idx_ckd]
            raw_htn = probs[idx_htn]
            raw_norm = probs[idx_norm]
            exp_values = np.exp([raw_ckd, raw_htn, raw_norm])
            percentages = exp_values / exp_values.sum() * 100
            st.subheader("Prediction Probabilities")
            st.write(f"**CKD**: `{percentages[0]:.1f}%`")
            st.write(f"**Hypertension**: `{percentages[1]:.1f}%`")
            st.write(f"**Normal**: `{percentages[2]:.1f}%`")

            st.subheader("Explainability – Saliency Map")
            disease_choice = st.selectbox("Choose a disease to visualise", display_diseases)
            if st.button("Generate Heatmap"):
                with st.spinner("Computing saliency map..."):
                    idx = full_target_columns.index(disease_choice)
                    heatmap = saliency_map(img_array, model, idx)
                    heatmap_resized = cv2.resize(heatmap, (original_img.width, original_img.height))
                    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                    superimposed = np.array(original_img) * 0.6 + heatmap_color * 0.4
                    superimposed = superimposed.astype(np.uint8)
                    st.image(superimposed, caption=f'Saliency map for {disease_choice}', width=400)

            if st.button("Download Report (PDF)"):
                with st.spinner("Generating report..."):
                    user_data = st.session_state.user_data
                    pdf_path = generate_pdf(user_data, probs, original_img, img_array, model)
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="Click to Download PDF",
                            data=f,
                            file_name="retinal_report.pdf",
                            mime="application/pdf"
                        )
                    os.remove(pdf_path)

    st.markdown("</div>", unsafe_allow_html=True)  # close dark container

st.markdown("</div>", unsafe_allow_html=True)  # close main-content

# ---------- Sticky Footer ----------
st.markdown("""
<div class='footer'>
    © 2025 CKD & Hypertension Predictor | AI for Non‑Invasive Screening
</div>
""", unsafe_allow_html=True)