import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------
# Title & Description
# -------------------------------
st.title("🧠 Human Activity Recognition App")
st.write("Upload an image to detect the human activity from 15 classes.")

# -------------------------------
# Load Model (Cached)
# -------------------------------
@st.cache_resource
def load_model_cached():
    model = load_model("har_model.h5")
    return model

model = load_model_cached()

# -------------------------------
# Class Labels
# -------------------------------
class_names = [
    'sitting',
    'using_laptop',
    'hugging',
    'sleeping',
    'drinking',
    'clapping',
    'dancing',
    'cycling',
    'calling',
    'laughing',
    'eating',
    'fighting',
    'listening_to_music',
    'running',
    'texting'
]

# -------------------------------
# Sidebar Info
# -------------------------------
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This app uses a Deep Learning model to classify human activities from images.\n\n"
    "Classes: 15 Activities"
)

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an activity image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Prediction Section
# -------------------------------
if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocessing
    img_resized = img.resize((224, 224))  # must match training size
    img_array = image.img_to_array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("🔍 Analyzing image..."):
        prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = np.max(prediction)

    # -------------------------------
    # Result Display
    # -------------------------------
    st.success(f"✅ Prediction: **{predicted_class}**")
    st.info(f"📊 Confidence: **{confidence:.2f}**")

    # Confidence Interpretation
    if confidence > 0.80:
        st.success("High confidence prediction ✅")
    elif confidence > 0.50:
        st.warning("Moderate confidence ⚠️")
    else:
        st.error("Low confidence ❌")

    # -------------------------------
    # Show All Probabilities
    # -------------------------------
    st.subheader("📊 Class Probabilities")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {prob:.2f}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("👩‍💻 Developed for Human Activity Recognition Project")
