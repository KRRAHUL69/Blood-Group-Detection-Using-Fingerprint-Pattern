from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import ssl

# Disable SSL verification for downloading pre-trained models, if needed
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model("./notebook/model.keras")

# Define the allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_path):
    """
    Preprocesses the image for model prediction.

    Args:
        file_path (str): Path to the image file.

    Returns:
        np.ndarray: Preprocessed image ready for predicion.
    """

    img = load_img(file_path, target_size=(64, 64))  # Resize image match the model's input size
    img_array = img_to_array(img) # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

# Endpoint to predict the blood group from fingerprint image
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = next(iter(request.files.values()), None)
        if not file:
            return jsonify({'error': 'No file provided'}), 400

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types are png, jpg, jpeg'}), 400

        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        try:
            # Preprocess the image
            img = preprocess_image(file_path)

            # Perform prediction
            prediction = model.predict(img)
            predicted_class = int(np.argmax(prediction[0]))
            print('predicted_class is : ', predicted_class)

            # Optimal: Define class names (if not in the model)
            class_names = ['A+', 'A-', 'B+', 'B-', 'AB+' 'AB-', 'O+', 'O-']
            predicted_label= class_names[predicted_class]

            # Return the result as JSON
            return jsonify({
                'name': request.form.get('name', 'undefined'),
                'phone': request.form.get('phone', 'undefined'),
                'gender': request.form.get('gender', 'undefined'),
                'age': request.form.get('age', 'undefined'),
                'prediction_class': predicted_class,
                'predicted_label': predicted_label,
                'confidence': float(np.max(prediction[0]))
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
        finally:
            # Clean up: remove the saved file
            if os.path.exists(file_path):
                os.remove(file_path)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
