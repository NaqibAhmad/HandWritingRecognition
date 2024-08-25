import numpy as np
import mlflow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
import streamlit as st

# Load the trained model
mlflow_model_uri = "D:/InternshipPractical/HandwritingRecognition/mlruns/181419095666103834/cda5062b1c9a4fda94638ffd0b0d3860/artifacts/model_mnist_cnn_final"  # Adjust to the correct model URI
model = mlflow.tensorflow.load_model(mlflow_model_uri)

# Function to preprocess the image
def preprocess_image(image_bytes):
    # Load image from bytes
    img = load_img(BytesIO(image_bytes), target_size=(28, 28), color_mode='grayscale')
    img_array = img_to_array(img)
    img_array = img_array.reshape(-1, 28, 28, 1)  # Reshape for model input
    img_array = img_array.astype('float32') / 255.0  # Normalize
    return img_array

# Streamlit app
st.title("Handwritten Digit Recognition")

# Upload image
uploaded_file = st.file_uploader("Choose a JPG or PNG image", type=["jpg", "png"])

if uploaded_file is not None:
    # Read file content
    image_bytes = uploaded_file.read()

    # Preprocess the image
    img_array = preprocess_image(image_bytes)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    # Display result
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write(f"Predicted Digit: {int(predicted_class[0])}")

