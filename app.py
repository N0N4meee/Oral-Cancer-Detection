import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)
MODEL_PATH = "best_model.h5"
THRESHOLD = 0.7

st.set_page_config(
    page_title="Oral Cancer Detection",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("Oral Cancer Detection")
st.write(
    "Upload an **oral histopathology image** to classify it as "
    "**Normal** or **Oral Squamous Cell Carcinoma (OSCC)**."
)

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Predict"):
        with st.spinner("Predicting..."):
            img_array = preprocess_image(image)
            prediction = model.predict(img_array)[0][0]

        prediction = float(prediction)

        st.write(f"**Raw prediction value:** `{prediction:.4f}`")

        if prediction < THRESHOLD:
            label = "Normal"
            confidence = 1 - prediction
            st.success(f"✅ Prediction: **{label}**")
        else:
            label = "Squamous Cell Carcinoma"
            confidence = prediction
            st.error(f"⚠️ Prediction: **{label}**")

        confidence = float(confidence) 
        st.info(f"Confidence: **{confidence:.2f}**")
        st.progress(confidence)

