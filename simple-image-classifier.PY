# simple-image-classifier
# A beginner AI image classifier using Python and MobileNetV2

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image
import numpy as np


def classify_image(image_path):
    # Load pre-trained AI model
    model = MobileNetV2(weights="imagenet")

    # Load image and resize it for the model
    img = image.load_img(image_path, target_size=(224, 224))

    # Convert image to array
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prepare image for MobileNetV2
    img_array = preprocess_input(img_array)

    # Make prediction
    predictions = model.predict(img_array)

    # Decode top 3 results
    results = decode_predictions(predictions, top=3)[0]

    print("\nImage Classification Results:")
    print("--------------------------------")

    for i, result in enumerate(results, start=1):
        label = result[1]
        confidence = result[2] * 100
        print(f"{i}. {label} - {confidence:.2f}%")


def main():
    print("===================================")
    print("      Simple Image Classifier")
    print("===================================")

    image_path = input("Enter image file path: ")

    try:
        classify_image(image_path)
    except Exception as e:
        print("\nError:")
        print("Could not classify the image.")
        print("Make sure the image path is correct.")
        print(e)


if __name__ == "__main__":
    main()
