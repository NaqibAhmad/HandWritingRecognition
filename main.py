import pandas as pd
import numpy as np
import mlflow
import tensorflow as tf
import mlflow.tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


# Set the experiment name in MLflow
mlflow.set_experiment("MNIST_CNN_Experiment_Final")

# Load the MNIST dataset
def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data() 
    return (X_train, y_train), (X_test, y_test)



(X_train, y_train), (X_test, y_test) = load_mnist()

# Reshape the data
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Normalize pixel values to be between 0 and 1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


# Define the CNN model architecture
def create_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(28, 28, 1), dtype='float32', name='input_layer')),
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Create a pipeline for preprocessing and model training
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),  # Scaling input features
])

# Start MLflow run
with mlflow.start_run():
    model = create_model()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_accuracy:.2f}')

    # Log metrics
    mlflow.log_metric("Test Accuracy", test_accuracy)
    mlflow.log_metric("Test Loss", test_loss)

    # Log the model
    mlflow.tensorflow.log_model(model, "model_mnist_cnn_final")