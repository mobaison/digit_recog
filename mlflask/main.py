from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
import io
import os

app = Flask(__name__)

# Load the TFLite model
model_path = os.path.join(os.path.dirname(__file__), "model2.tflite")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input & output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image):
    """Preprocess image for model prediction"""
    img = cv2.resize(image, (28, 28))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # Reshape for model
    return img

@app.route('/')
def index():
    return render_template('index.html')

def predict(image):
    """Runs inference using TensorFlow Lite"""
    processed_img = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], processed_img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return int(np.argmax(output_data))

@app.route('/predict_canvas', methods=['POST'])
def predict_canvas():
    data = request.json['image']
    img_data = base64.b64decode(data.split(',')[1])
    img = Image.open(io.BytesIO(img_data)).convert("L")
    img = np.array(img)
    predicted_label = predict(img)
    return jsonify({'prediction': predicted_label})

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    img = Image.open(file).convert("L")
    img = img.resize((28, 28))
    img = np.array(img)
    predicted_label = predict(img)
    return jsonify({'prediction': predicted_label})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
