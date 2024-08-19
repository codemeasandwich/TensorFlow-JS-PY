import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

print("OS:", os.name)
print("NumPy:", np.__version__)
print("TensorFlow:", tf.__version__)

# Set the path to the directory containing the MNIST dataset
data_dir = '../dataset'

# Load the training images and labels
train_images_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
train_labels_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
with open(train_images_path, 'rb') as f:
    train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
with open(train_labels_path, 'rb') as f:
    train_labels = np.frombuffer(f.read(), np.uint8, offset=8)

# Load the test images and labels
test_images_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')
test_labels_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')
with open(test_images_path, 'rb') as f:
    test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
with open(test_labels_path, 'rb') as f:
    test_labels = np.frombuffer(f.read(), np.uint8, offset=8)

# Normalize pixel values
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# Add channel dimension
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# One-hot encode labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Print the shapes of the loaded data
print("Training images shape:", train_images.shape)
print("Training labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)

from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import time
import numpy as np

# ================================
# ======== BENCHMARKING ==========
# ================================

def benchmark(func, *args, **kwargs):
    """
    Decorator function to measure the execution time of a given function.
    Prints the function name and the time taken for execution in milliseconds.
    """
    print(f"Starting {func.__name__}...")
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"{func.__name__} completed in {(end_time - start_time) * 1000:.2f} milliseconds.")
    return result

# ================================
# ======== PREPROCESSING =========
# ================================

def normalize_peek(images, text="?"):
    """
    Utility function to check the range of values across the entire dataset.
    Prints the minimum and maximum values of the images.
    """
    min_value = np.min(images)
    max_value = np.max(images)
    print(f"{text} normalization min:{min_value} / max:{max_value}")

def preprocess_data(train_images, test_images, train_labels, test_labels):
    """
    Preprocesses the training and test data.
    Normalizes the pixel values and reshapes the images.
    """
    # Normalize pixel values
    normalize_peek(train_images, "Train Before")
    train_images = train_images / 255
    normalize_peek(train_images, "Train After")

    normalize_peek(test_images, "Test Before")
    test_images = test_images / 255
    normalize_peek(test_images, "Test After")

    # Reshape the images
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    return train_images, test_images, train_labels, test_labels

# ================================
# ======== TRAINING ==============
# ================================

def create_model():
    """
    Creates the CNN model architecture.
    """
    model = keras.Sequential([
        layers.Conv2D(30, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        keras.layers.Dense(10, activation='sigmoid')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def app():
    """
    Main function that orchestrates the entire flow of the application.
    Loads the MNIST dataset, preprocesses the data, creates the model,
    trains the model, and evaluates its performance.
    """
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = benchmark(keras.datasets.mnist.load_data)

    # Preprocess the data
    preprocessed_train_images, preprocessed_test_images, preprocessed_train_labels, preprocessed_test_labels = benchmark(
        preprocess_data, train_images, test_images, train_labels, test_labels)

    # Create the model
    model = benchmark(create_model)

    # Train the model
    benchmark(model.fit, preprocessed_train_images, preprocessed_train_labels, epochs=5, batch_size=32)

    # Evaluate the model on test data
    def evaluate_model():
        test_loss, test_accuracy = model.evaluate(preprocessed_test_images, preprocessed_test_labels)
        print("Test Loss:", test_loss)
        print("Test Accuracy:", test_accuracy)

    benchmark(evaluate_model)

    return {
        'training': {
            'images': preprocessed_train_images,
            'labels': preprocessed_train_labels
        },
        'testing': {
            'images': preprocessed_test_images,
            'labels': preprocessed_test_labels
        },
        'model': model
    }

preprocessed = benchmark(app)

# ================================
# ======== TESTING ===============
# ================================

def test_image(preprocessed, index):
    """
    Tests the trained model on a specific image from the test set.
    Prints the predicted label and the true label for the image.
    """
    test_images = preprocessed['testing']['images']
    test_labels = preprocessed['testing']['labels']
    model = preprocessed['model']

    # Get the image at the selected index
    test_image = test_images[index]

    print("Making predictions on the selected image...")
    predictions = model.predict(test_image.reshape((1, 28, 28, 1)))
    print(predictions)
    # Get the predicted label
    predicted_label = np.argmax(predictions)

    # Get the true label
    true_label = test_labels[index]

    print(f"Predicted Label: {predicted_label}, Image Label: {true_label}, INDEX: {index}")

# Select an index of the image to test
test_image_index = 1869

# Test the image at the selected index
benchmark(test_image, preprocessed, test_image_index)