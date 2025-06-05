import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
from PIL import Image, ImageDraw
import io
import os
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import MultiHeadAttention

model = tf.keras.models.load_model(
    "fusion_model_best.h5",
    custom_objects={'MultiHeadAttention': MultiHeadAttention}
)

# Load MobileNetV2 for drawing feature extraction
mobilenet_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, pooling='avg', weights='imagenet')

# Load EfficientNetB0 for MRI feature extraction
efficientnet_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, pooling='avg', weights='imagenet')

# App Title
st.title("🧠 Parkinson's Disease & Stage Classifier")
st.markdown("Upload patient inputs across modalities to detect PD and its stage.")

# Helper: Extract MFCCs from .wav
def extract_mfcc(file, n_mfcc=22):
    y, sr = librosa.load(file, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

# Helper: Convert spiral .txt to image
def spiral_txt_to_image(txt_file):
    content = txt_file.read().decode()
    coords = [list(map(int, line.strip().split(';')[:2])) for line in content.strip().split('\n') if ';' in line]
    img = Image.new("RGB", (300, 300), "white")
    draw = ImageDraw.Draw(img)
    for x, y in coords:
        draw.ellipse((x, y, x+2, y+2), fill="black")
    img = img.resize((224, 224)).convert("RGB")
    return img

# Helper: Extract embedding from image
def extract_embedding(img, model, preprocess_func):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_func(img_array)
    return model.predict(img_array)[0]

# --- Voice Input ---
st.header("🎙️ Voice Input")
voice_source = st.radio("Select Voice Input Type", ["Upload .wav file", "Enter UCI voice features manually"])

voice_vector = None
if voice_source == "Upload .wav file":
    wav_file = st.file_uploader("Upload patient's voice (.wav)", type=["wav"])
    if wav_file:
        voice_vector = extract_mfcc(wav_file)
elif voice_source == "Enter UCI voice features manually":
    st.markdown("Enter 22 comma-separated voice features")
    voice_str = st.text_input("UCI Voice Features (22 floats)")
    if voice_str:
        try:
            voice_vector = np.array([float(x.strip()) for x in voice_str.split(",")])
        except:
            st.error("Invalid format. Ensure you provide exactly 22 float values.")

# --- Drawing Input ---
st.header("✍️ Drawing Input")
drawing_file = st.file_uploader("Upload spiral drawing (.txt, .png, .jpg)", type=["txt", "png", "jpg", "jpeg"])
drawing_vector = None
if drawing_file:
    try:
        if drawing_file.name.endswith(".txt"):
            img = spiral_txt_to_image(drawing_file)
        else:
            img = Image.open(drawing_file).resize((224, 224)).convert("RGB")
        drawing_vector = extract_embedding(img, mobilenet_model, mobilenet_preprocess)
    except Exception as e:
        st.error(f"Drawing preprocessing error: {e}")

# --- MRI Input ---
st.header("🧠 MRI Input")
mri_file = st.file_uploader("Upload MRI image (.png)", type=["png", "jpg", "jpeg"])
mri_vector = None
if mri_file:
    try:
        img = Image.open(mri_file).resize((224, 224)).convert("RGB")
        mri_vector = extract_embedding(img, efficientnet_model, efficientnet_preprocess)
    except Exception as e:
        st.error(f"MRI preprocessing error: {e}")

# --- Symptom Checklist ---
st.header("📋 Symptom Checklist")
symptom_text = open("Parkinson symptoms.txt", "r").readlines()
symptom_list = [line.strip() for line in symptom_text if line.strip() and not any(x in line for x in ["Motor", "Non", ":", "🔴", "🟡", "🟢"])]
selected_symptoms = st.multiselect("Select observed symptoms", symptom_list)
symptom_vector = np.array([1 if s in selected_symptoms else 0 for s in symptom_list])

# --- Predict Button ---
if st.button("🧪 Predict"):
    if any(v is None for v in [voice_vector, drawing_vector, mri_vector]) or symptom_vector.shape[0] != 31:
        st.error("Please make sure all modalities are uploaded and valid.")
    else:
        # Prepare batch
        voice_input = np.expand_dims(voice_vector, axis=0)
        drawing_input = np.expand_dims(drawing_vector, axis=0)
        mri_input = np.expand_dims(mri_vector, axis=0)
        symptom_input = np.expand_dims(symptom_vector, axis=0)

        # Predict
        pred_pd, pred_stage = model.predict([voice_input, drawing_input, mri_input, symptom_input])
        pred_pd_label = "Parkinson's Disease" if pred_pd[0][0] > 0.5 else "Healthy"
        pred_stage_label = ["Early", "Mid", "Late"][np.argmax(pred_stage[0])]

        # Output
        st.success(f"🧠 Diagnosis: **{pred_pd_label}**")
        if pred_pd_label == "Parkinson's Disease":
            st.info(f"📊 Stage: **{pred_stage_label}**")
