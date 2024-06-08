import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from .models import get_disease_model

# Load the model once at the module level
model = get_disease_model()

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def predict_disease(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    # Assuming you have a list of class names in order of the model's output
    class_names = ['Pepper__bell___Bacterial_spot', 'Tomato_healthy', ...]  # replace with actual class names
    predicted_class = class_names[np.argmax(prediction)]

    # Provide precautions based on the predicted class
    precautions = get_precautions(predicted_class)
    return predicted_class, precautions

def get_precautions(disease):
    precautions = {
        'Pepper__bell___Bacterial_spot': 'Effective management of pepper bacterial spot involves a combination of cultural practices, resistant varieties, chemical treatments, biological controls, and integrated pest management. Regular monitoring and early intervention are crucial for controlling the disease and minimizing its impact on pepper crops',
        'Class2': 'Precautions for Class2...',
        # Add other class precautions as needed
    }
    return precautions.get(disease, 'No precautions available')
