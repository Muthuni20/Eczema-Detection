import os
import numpy as np
import tensorflow.keras as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import tempfile

app = Flask(__name__)
CORS(app)

# Specify the location of your trained model
model = K.models.load_model("D:/allergydetection/second/model.h5")

# Define the class names for your specific classes (eczema)
types = ['eczema', 'normal_skin']  # Update with your class names

# Define the probability threshold
probability_threshold = 0.45

# Function to make predictions
def predict(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = img_to_array(img)
    x = K.applications.xception.preprocess_input(x)

    prediction = model.predict(np.array([x]))[0]
    detected_class_index = np.argmax(prediction)
    detected_class_name = types[detected_class_index]
    detected_class_probability = prediction[detected_class_index]

    if detected_class_probability >= probability_threshold:
        return detected_class_name, detected_class_probability * 100.0
    else:
        return "normal_skin", detected_class_probability * 100.0

# Function to check if the image is suitable for classification as a skin image
def is_skin_image(image):
    # Convert the image to the HSV color space
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for the skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask to extract skin regions
    skin_mask = cv2.inRange(hsv_img, lower_skin, upper_skin)

    # Calculate the percentage of skin pixels in the image
    total_pixels = image.shape[0] * image.shape[1]
    skin_pixels = np.sum(skin_mask == 255)
    skin_percentage = (skin_pixels / total_pixels) * 100.0

    # You can adjust the threshold based on your requirements
    skin_threshold = 1.0  # Adjust this threshold as needed

    # Check if the percentage of skin pixels exceeds the threshold
    if skin_percentage >= skin_threshold:
        return True
    else:
        return False

# Define a route to upload and predict an image
@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Create a temporary file to save the uploaded image
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file.save(temp_file.name)

        # Read the temporary image file using OpenCV
        image_data = cv2.imread(temp_file.name)

    if is_skin_image(image_data):
        detected_class, probability = predict(temp_file.name)
        return jsonify({
            'detectedClass': detected_class,
            'probability': probability,
        })
    else:
        return jsonify({'error': 'Invalid image'})

if __name__ == '__main__':
    app.run(debug=True)
