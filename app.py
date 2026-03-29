import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load your model
model = load_model('fruit_ripeness_model.h5')

# Define your class labels
class_names = ['Raw', 'Ripe']

st.title("🍎 Fruit Ripeness Detector")
st.write("Upload a fruit image to check if it's ripe or raw.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img = image.resize((128, 128))  # Match your model's input size
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"🍌 Prediction: **{predicted_class}**")
