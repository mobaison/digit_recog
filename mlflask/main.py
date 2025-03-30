from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import base64
import io
import os

app = Flask(__name__)
model = load_model("model2.h5")  # Ensure the correct model path

def preprocess_image(image):
    """Preprocess image for model prediction"""
    img = cv2.resize(image, (28, 28))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_canvas', methods=['POST'])
def predict_canvas():
    data = request.json['image']
    img_data = base64.b64decode(data.split(',')[1])
    img = Image.open(io.BytesIO(img_data)).convert("L")
    img = np.array(img)
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    predicted_label = int(np.argmax(prediction))
    return jsonify({'prediction': predicted_label})

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    img = Image.open(file).convert("L")
    img = img.resize((28, 28))  # Ensure correct resizing
    img = np.array(img)
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    predicted_label = int(np.argmax(prediction))
    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
