🧠  Parkinson's Multimodal Detection & Stage Classification

This Streamlit app allows real-time, multimodal prediction of Parkinson's Disease (PD) and its clinical stage (Early, Mid, Late) using voice, drawing, MRI, and symptom checklist inputs.

🚀 Features

Upload or input voice data: .wav file or UCI features

Upload drawing: .txt spiral file or image (.png/.jpg)

Upload MRI image: .png/.jpg

Select observed symptoms via checkbox list

Outputs:

✅ PD vs Healthy classification

📊 Stage prediction (Early, Mid, Late) if PD is detected

💡 Technology Stack

Streamlit for web app interface

TensorFlow/Keras for deep learning model

librosa for voice MFCC extraction

MobileNetV2 for drawing feature embeddings

EfficientNetB0 for MRI embeddings

Self-Attention fusion model architecture

📂 Files in This Repo

File

Description

pd_app.py

Main Streamlit app

fusion_model_best.h5

Trained self-attention fusion model

Parkinson symptoms.txt

List of symptoms used in the checklist

requirements.txt

Python dependencies for Streamlit deployment

🧪 Try It Online

You can run this app on Streamlit Cloud by deploying this repo and selecting pd_app.py.

🧰 How to Run Locally

Clone this repo:

git clone https://github.com/yourusername/parkinsons-multimodal-detection.git
cd parkinsons-multimodal-detection

Install dependencies:

pip install -r requirements.txt

Launch the app:

streamlit run pd_app.py

📎 Sample Inputs (Optional)

To test the app, prepare:

A .wav voice file (or UCI feature vector)

A spiral .txt drawing or .png image

An MRI brain scan image (.png)

👩‍⚕️ Clinical Relevance

This project enables AI-powered stage-wise Parkinson’s detection using multimodal real-world data and is deployable on mobile/web via Streamlit.
