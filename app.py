import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your trained model (replace with your actual model path)
MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)

# Define class names based on your model training
class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut', 'Planthopper', 'Rice hispa', 'Steam borer whiteheads']

# Function to preprocess the uploaded image
def load_and_prep_image(img, img_size=128):
    img = img.resize((img_size, img_size))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit app layout
st.title("Image Processing Technique of Rice Plants")
st.write("Upload an image of a rice plant, and this tool will classify whether it's a disease or pest attack.")

# Image uploader
uploaded_file = st.file_uploader("Please Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Add the "Predict" button
    if st.button('Predict'):
        st.write("Classifying...")

        # Preprocess the image and make a prediction
        img_array = load_and_prep_image(img)
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]

        # Display the result
        st.write(f"Predicted class: **{predicted_class}**")
