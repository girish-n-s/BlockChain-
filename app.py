from flask import Flask, render_template, request, redirect, url_for
#import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
import cv2


app = Flask(__name__)

# Load TensorFlow Model
#model = tf.keras.models.load_model("trained_model.keras")

# Define classes
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Prediction Function


def detect_leaf(image_path):
    """
    Detects if a leaf is present in the given image.
    
    Args:
        image_path (str): Path to the input image file.
    
    Returns:
        bool: True if a leaf is detected, False otherwise.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a broader range for green color detection
    lower_green = np.array([20, 30, 20])  # Relaxed lower bound
    upper_green = np.array([90, 255, 255])  # Allow brighter greens
    
    # Create a mask for green areas
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small gaps

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Analyze contours
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 800:  # Larger threshold for leaf-like regions
            # Check shape using convexity or aspect ratio
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            width = rect[1][0]
            height = rect[1][1]
            
            # Avoid elongated shapes that are unlikely to be leaves
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)
                if 1.0 < aspect_ratio < 3.5:  # Acceptable range for leaf-like shapes
                    print("Leaf detected! Proceeding to disease classification.")
                    return True

    print("No leaf detected! Skipping disease classification.")
    return False
def model_prediction(image_path):
    if(detect_leaf(image_path)):
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to batch
        predictions = model.predict(input_arr)
        return class_names[np.argmax(predictions)]
    else :
        return "image error"

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/disease', methods=['GET', 'POST'])
def disease_recognition():
    if request.method == 'POST':
        # Handle file upload
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        
        # Run prediction
        prediction = model_prediction(file_path)
        return render_template('disease.html', prediction=prediction)
    
    return render_template('disease.html')

if __name__ == '__main__':
    # Ensure uploads folder exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)