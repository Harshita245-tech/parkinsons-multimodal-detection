import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
from PIL import Image, ImageDraw
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import (
    Dense, Dropout, Lambda, Concatenate,
    GlobalAveragePooling1D, MultiHeadAttention
)
from tensorflow.keras.models import load_model

# Define input expanders
def expand_dims(x):
    return tf.expand_dims(x, axis=1)

# Register custom layers
custom_objects = {
    "expand_dims": expand_dims,
    "Dense": Dense,
    "Dropout": Dropout,
    "Lambda": Lambda,
    "Concatenate": Concatenate,
    "GlobalAveragePooling1D": GlobalAveragePooling1D,
    "MultiHeadAttention": MultiHeadAttention
}

# Load trained fusion model
model = load_model("fusion_model_best.keras", custom_objects=custom_objects)

# Load CNNs for embedding extraction
mobilenet_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, pooling='avg', weights='imagenet')
efficientnet_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, pooling='avg', weights='imagenet')

# Streamlit UI
st.set_page_config(page_title="Parkinson’s Diagnosis", layout="centered")
st.title("🧠 Parkinson's Disease & Stage Classifier")
st.markdown("Upload inputs across modalities to detect PD and its stage.")

# Extract MFCCs
def extract_mfcc(file, n_mfcc=22):
    y, sr = librosa.load(file, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

# Spiral .txt to image
def spiral_txt_to_image(txt_file):
    content = txt_file.read().decode()
    coords = [list(map(int, line.strip().split(';')[:2])) for line in content.strip().split('\n') if ';' in line]
    img = Image.new("RGB", (300, 300), "white")
    draw = ImageDraw.Draw(img)
    for x, y in coords:
        draw.ellipse((x, y, x + 2, y + 2), fill="black")
    return img.resize((224, 224)).convert("RGB")

# Extract image embedding
def extract_embedding(img, model, preprocess_func):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_func(img_array)
    return model.predict(img_array, verbose=0)[0]

# 🎙️ Voice input
st.header("🎙️ Voice Input")
voice_vector = None
voice_source = st.radio("Input type:", ["Upload .wav file", "Enter UCI features"])
if voice_source == "Upload .wav file":
    wav_file = st.file_uploader("Upload voice (.wav)", type=["wav"])
    if wav_file:
        voice_vector = extract_mfcc(wav_file)
else:
    voice_str = st.text_input("Enter 22 UCI features (comma-separated):")
    if voice_str:
        try:
            voice_vector = np.array([float(v.strip()) for v in voice_str.split(",")])
        except:
            st.error("⚠ Invalid format. Use 22 comma-separated floats.")

# ✍️ Drawing input
st.header("✍️ Drawing Input")
drawing_vector = None
drawing_file = st.file_uploader("Upload spiral drawing (.txt or image)", type=["txt", "png", "jpg", "jpeg"])
if drawing_file:
    try:
        if drawing_file.name.endswith(".txt"):
            img = spiral_txt_to_image(drawing_file)
        else:
            img = Image.open(drawing_file).resize((224, 224)).convert("RGB")
        drawing_vector = extract_embedding(img, mobilenet_model, mobilenet_preprocess)
    except Exception as e:
        st.error(f"Drawing error: {e}")

# 🧠 MRI input
st.header("🧠 MRI Input")
mri_vector = None
mri_file = st.file_uploader("Upload MRI image (.png/.jpg)", type=["png", "jpg", "jpeg"])
if mri_file:
    try:
        img = Image.open(mri_file).resize((224, 224)).convert("RGB")
        mri_vector = extract_embedding(img, efficientnet_model, efficientnet_preprocess)
    except Exception as e:
        st.error(f"MRI error: {e}")

# 📋 Symptom input
st.header("📋 Symptoms")
symptom_lines = open("Parkinson symptoms.txt").readlines()
symptom_list = [line.strip() for line in symptom_lines if line.strip() and not any(x in line for x in ["Motor", "Non", ":", "🔴", "🟡", "🟢"])]
selected_symptoms = st.multiselect("Select symptoms", symptom_list)
symptom_vector = np.array([1 if s in selected_symptoms else 0 for s in symptom_list])

# --- Predict button ---
if st.button("🧪 Predict"):
    if any(v is None for v in [voice_vector, drawing_vector, mri_vector]) or symptom_vector.shape[0] != 31:
        st.error("⚠ Please ensure all modalities are correctly uploaded.")
    else:
        inputs = [
            np.expand_dims(voice_vector, axis=0),
            np.expand_dims(drawing_vector, axis=0),
            np.expand_dims(mri_vector, axis=0),
            np.expand_dims(symptom_vector, axis=0),
        ]
        pred_pd, pred_stage = model.predict(inputs)
        pd_result = "Parkinson's Disease" if pred_pd[0][0] > 0.5 else "Healthy"
        stage_result = ["Early", "Mid", "Late"][np.argmax(pred_stage[0])]

        st.success(f"🧠 Prediction: **{pd_result}**")
        if pd_result == "Parkinson's Disease":
            st.info(f"📊 Stage: **{stage_result}**")
