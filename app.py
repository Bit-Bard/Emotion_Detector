import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2

# Load the pre-trained model
try:
    model = load_model("model_full.keras")  # Use "model_full.h5" if saved in HDF5 format
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
facec = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotion(image):
    try:
        gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_image, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_image[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]
            return pred
    except Exception as e:
        st.error(f"Error detecting emotion: {e}")
        return "Unable to detect"

st.title("Emotion Detector")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    st.write("Classifying...")
    emotion = detect_emotion(image)
    st.write(f"Detected Emotion: {emotion}")