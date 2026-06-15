import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import h5py
import json

# Set premium page configs
st.set_page_config(
    page_title="PneuSight AI - Chest X-Ray Analyzer",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-end aesthetics (dark theme, glassmorphism, responsive elements, animations)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* Global Reset & Styling */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Custom layout background adjustments */
.stApp {
    background-color: #090d16;
}

/* Header Banner Styling */
.header-container {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
    padding: 3rem 2rem;
    border-radius: 24px;
    border: 1px solid rgba(99, 102, 241, 0.2);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
    margin-bottom: 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.header-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.12) 0%, transparent 60%);
    pointer-events: none;
}

.app-title {
    background: linear-gradient(90deg, #38bdf8, #818cf8, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 3rem;
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
}

.app-subtitle {
    color: #94a3b8;
    font-size: 1.2rem;
    font-weight: 400;
    max-width: 700px;
    margin: 0 auto;
    line-height: 1.6;
}

/* Glassmorphic Cards */
.glass-card {
    background: rgba(15, 23, 42, 0.55);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.35);
    transition: all 0.3s ease;
}

.glass-card:hover {
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: 0 15px 40px rgba(99, 102, 241, 0.12);
    transform: translateY(-2px);
}

.glass-card-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 0.75rem;
}

/* Results Section styling */
.result-box {
    padding: 1.75rem;
    border-radius: 16px;
    margin-top: 1rem;
    border: 1px solid;
    animation: slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.normal-bg {
    background: rgba(16, 185, 129, 0.08);
    border-color: rgba(16, 185, 129, 0.25);
    color: #34d399;
}

.pneumonia-bg {
    background: rgba(239, 68, 68, 0.08);
    border-color: rgba(239, 68, 68, 0.25);
    color: #f87171;
}

.badge {
    padding: 0.5rem 1.25rem;
    border-radius: 9999px;
    font-weight: 800;
    text-transform: uppercase;
    font-size: 1rem;
    letter-spacing: 1px;
    display: inline-block;
    margin-bottom: 1rem;
}

.badge-normal {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: #ffffff;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
}

.badge-pneumonia {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    color: #ffffff;
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
}

.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.stat-row:last-child {
    border-bottom: none;
}

.stat-label {
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 500;
}

.stat-value {
    color: #f1f5f9;
    font-size: 1rem;
    font-weight: 600;
}

/* File Uploader styling custom overrides */
div[data-testid="stFileUploader"] {
    background: rgba(30, 41, 59, 0.25);
    border: 2px dashed rgba(99, 102, 241, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    transition: all 0.3s ease;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #38bdf8;
    background: rgba(30, 41, 59, 0.4);
}
</style>
""", unsafe_allow_html=True)


# Helper function to clean the H5 model config (handles Keras 3 deserialization errors)
def fix_model_config(input_path, output_path):
    def clean_config(config):
        if isinstance(config, dict):
            config.pop('quantization_config', None)
            for k, v in list(config.items()):
                clean_config(v)
        elif isinstance(config, list):
            for item in config:
                clean_config(item)

    try:
        with h5py.File(input_path, 'r') as f_in:
            with h5py.File(output_path, 'w') as f_out:
                for key in f_in.keys():
                    f_in.copy(key, f_out)
                for name, value in f_in.attrs.items():
                    if name == 'model_config':
                        config_str = value.decode('utf-8') if isinstance(value, bytes) else value
                        config_dict = json.loads(config_str)
                        clean_config(config_dict)
                        f_out.attrs[name] = json.dumps(config_dict).encode('utf-8')
                    else:
                        f_out.attrs[name] = value
        return True
    except Exception as e:
        st.error(f"Error during model conversion: {e}")
        return False


# Cache model loading to avoid reloading on every run
@st.cache_resource
def load_pneumonia_model():
    original_model_path = "pneumonia_cnn_model.h5"
    fixed_model_path = "pneumonia_cnn_model_fixed.h5"
    
    # Check if the original model exists
    if not os.path.exists(original_model_path):
        st.error(f"Model file '{original_model_path}' not found in the workspace directory. Please make sure it's placed in this directory.")
        return None
    
    # Try loading the model directly (it might be already cleaned or the environment supports it)
    try:
        model = tf.keras.models.load_model(original_model_path, compile=False)
        return model
    except Exception:
        # If loading original fails (due to quantization_config or other Keras 3 config errors)
        # Use the fixed version if it exists, otherwise fix it
        if not os.path.exists(fixed_model_path):
            with st.spinner("Optimizing model config for compatibility..."):
                success = fix_model_config(original_model_path, fixed_model_path)
                if not success:
                    return None
        
        try:
            model = tf.keras.models.load_model(fixed_model_path, compile=False)
            return model
        except Exception as e:
            st.error(f"Failed to load the model: {e}")
            return None


# Load model
model = load_pneumonia_model()

# Header Dashboard Banner
st.markdown("""
<div class="header-container">
    <div class="app-title">🩻 PneuSight AI</div>
    <div class="app-subtitle">Deep Learning Chest X-Ray Diagnostic Assistant. Upload a chest radiograph to evaluate and classify for potential signs of Pneumonia.</div>
</div>
""", unsafe_allow_html=True)

# Main Application Layout
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card-title">📁 Upload Radiograph</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload a Chest X-ray image (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )
    
    st.markdown("""
    <div style="margin-top: 1.5rem; font-size: 0.85rem; color: #64748b; line-height: 1.5;">
        <strong>Instructions:</strong><br>
        1. Select a high-quality anteroposterior (AP) chest X-ray.<br>
        2. Ensure the image is clear and cropped to focus primarily on the lung fields.<br>
        3. PNG, JPG, or JPEG formats are supported.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="glass-card-title">🔍 Image Preview</div>', unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card-title">📊 Analysis & Diagnosis</div>', unsafe_allow_html=True)
    
    if uploaded_file is None:
        st.info("Awaiting X-Ray image upload to start analysis.")
    else:
        if model is None:
            st.error("Model could not be loaded. Please ensure the model file is valid.")
        else:
            with st.spinner("Analyzing image..."):
                # Preprocess image
                # Model expects (None, 128, 128, 3) input shape
                img = image.convert("RGB")
                img = img.resize((128, 128))
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Make prediction
                prediction = model.predict(img_array)[0][0]
                
                # In binary classification:
                # Typically, class 1 = Pneumonia, class 0 = Normal
                # Output sigmoid gives probability of class 1 (Pneumonia)
                is_pneumonia = prediction >= 0.5
                confidence = prediction if is_pneumonia else (1.0 - prediction)
                confidence_percent = confidence * 100
                
                # Diagnosis display
                if is_pneumonia:
                    badge_class = "badge-pneumonia"
                    bg_class = "pneumonia-bg"
                    status_text = "Pneumonia Detected"
                else:
                    badge_class = "badge-normal"
                    bg_class = "normal-bg"
                    status_text = "Normal (No Pneumonia)"
                
                st.markdown(f"""
                <div class="result-box {bg_class}">
                    <span class="badge {badge_class}">{status_text}</span>
                    <h3 style="margin: 0.5rem 0 1rem 0; font-size: 1.6rem; font-weight: 700;">
                        Confidence: {confidence_percent:.2f}%
                    </h3>
                    <div style="font-size: 0.95rem; line-height: 1.6; color: #cbd5e1;">
                        The model identified patterns in the lung fields that match the visual criteria of 
                        <strong>{"Pneumonia" if is_pneumonia else "Normal healthy lung tissue"}</strong> with a confidence score of 
                        <strong>{confidence_percent:.2f}%</strong>.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### Prediction Probabilities")
                
                # Normal Probability Bar
                normal_prob = (1.0 - prediction) * 100
                st.write(f"Normal: {normal_prob:.1f}%")
                st.progress(normal_prob / 100.0)
                
                # Pneumonia Probability Bar
                pneumonia_prob = prediction * 100
                st.write(f"Pneumonia: {pneumonia_prob:.1f}%")
                st.progress(pneumonia_prob / 100.0)
                
                st.markdown("<br>", unsafe_allow_html=True)
                # Additional File details
                st.markdown("##### Metadata details")
                st.markdown(f"""
                <div class="stat-row">
                    <span class="stat-label">File Name</span>
                    <span class="stat-value">{uploaded_file.name}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Original Dimensions</span>
                    <span class="stat-value">{image.size[0]}x{image.size[1]} px</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Processed Resolution</span>
                    <span class="stat-value">128x128 px (RGB)</span>
                </div>
                """, unsafe_allow_html=True)
                
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar layout details
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0;">
        <h2 style="color: #38bdf8; font-weight: 800; margin-bottom: 0.25rem;">🔬 PneuSight AI</h2>
        <span style="color: #64748b; font-size: 0.9rem;">Model Info & Settings</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🧠 Neural Network Info")
    st.markdown("""
    - **Architecture:** Sequential Convolutional Neural Network (CNN)
    - **Expected Input Shape:** `(128, 128, 3)`
    - **Parameters:** 926,755
    - **Output Layer:** Sigmoid activation (Binary)
    - **Primary Model Weight File:** `pneumonia_cnn_model.h5`
    """)
    
    st.markdown("---")
    
    st.markdown("### ⚠️ Clinical Disclaimer")
    st.warning("""
    This application is an AI-based decision-support tool. It has been built and tested for educational purposes and is not a replacement for professional clinical evaluation, medical advice, diagnosis, or treatment. 
    
    Always consult a qualified radiologist or medical professional to interpret chest radiographs.
    """)
