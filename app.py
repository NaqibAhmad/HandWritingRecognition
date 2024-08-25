import numpy as np
import mlflow
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO

# Load the trained model
mlflow_model_uri = "mlruns/181419095666103834/9b746eaae2224c1a96ca83e46f751625/artifacts/model_mnist_cnn_final"  # Adjust to the correct model URI
model = mlflow.tensorflow.load_model(mlflow_model_uri)

# Create FastAPI app
app = FastAPI(
    title="Hand Writing Recognition Project",
)

# Define a prediction function
def preprocess_image(image_bytes):
    # Load image from bytes
    img = load_img(BytesIO(image_bytes), target_size=(28, 28), color_mode='grayscale')
    img_array = img_to_array(img)
    img_array = img_array.reshape(-1, 28, 28, 1)  # Reshape for model input
    img_array = img_array.astype('float32') / 255.0  # Normalize
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        return JSONResponse(content={"error": "File format not supported. Please upload a JPG or PNG file."}, status_code=400)
    
    # Read file content
    image_bytes = await file.read()

    # Preprocess the image
    img_array = preprocess_image(image_bytes)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    return JSONResponse(content={"predicted_digit": int(predicted_class[0])})

# Run the app using: uvicorn filename:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload = True)