import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("/Users/mobaisonoinam/Documents/scikit/ml_train/model2.h5")  

# Canvas settings
drawing = False  # True when mouse is pressed
ix, iy = -1, -1  # Initial mouse coordinates
canvas = np.zeros((280, 280, 1), dtype=np.uint8)  # Black canvas

# Mouse callback function
def draw_digit(event, x, y, flags, param):
    global ix, iy, drawing, canvas

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(canvas, (x, y), 10, (255, 255, 255), -1)  # Draw white circles

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Create drawing window
cv2.namedWindow("Draw a Digit")
cv2.setMouseCallback("Draw a Digit", draw_digit)

while True:
    cv2.imshow("Draw a Digit", canvas)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):  # Press 's' to save and predict
        break
    elif key == ord("c"):  # Press 'c' to clear canvas
        canvas[:] = 0

cv2.destroyAllWindows()

# Preprocess the drawn image
def preprocess_image(image):
    img = cv2.resize(image, (28, 28))  # Resize to 28x28
    img = img.astype("float32") / 255.0  # Normalize
    img = np.expand_dims(img, axis=(0, -1))  # Add batch and channel dimensions
    return img

# Prepare image for model
processed_img = preprocess_image(canvas)

# Predict the drawn digit
prediction = model.predict(processed_img)
predicted_label = np.argmax(prediction)

# Display the prediction result
plt.imshow(canvas, cmap="gray")
plt.title(f"Predicted Digit: {predicted_label}")
plt.show()