import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('models/plant_disease_model_final.keras')

# Define a function to predict the disease
def predict_disease(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
    # Map the predicted class to the disease name
    class_names = ['Pepper__bell___Bacterial_spot', 'Tomato_healthy', 'Class3', ...]  # Replace with your actual class names
    disease_name = class_names[predicted_class]
    
    return disease_name
