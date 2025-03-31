import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Load the saved model
model = tf.keras.models.load_model("/Users/mobaisonoinam/Documents/scikit/ml_club/model2.h5")

def preprocess_image(image_path):
    """Process images for MNIST prediction"""
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to MNIST dimensions
        
        # Convert to numpy array and normalize
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Auto-invert colors if background is white
        if np.mean(img_array) > 0.5:
            img_array = 1 - img_array
            
        return img_array.reshape(1, 28, 28, 1)  # Add batch/channel dims
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def main():
    while True:
        print("\n" + "="*40)
        image_path = input("Enter image path (or 'q' to quit): ").strip()
        
        if image_path.lower() in ['q', 'quit']:
            break
            
        if not os.path.exists(image_path):
            print("File not found!")
            continue
            
        processed_image = preprocess_image(image_path)
        if processed_image is None:
            continue
            
        prediction = model.predict(processed_image)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        print(f"\nPrediction: {digit}")
        print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    print("MNIST Digit Classifier")
    print("======================")
    main()