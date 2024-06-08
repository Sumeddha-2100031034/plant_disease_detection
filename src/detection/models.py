import tensorflow as tf
import os

def get_disease_model():
    # The model is located in the 'models' directory at the root level of the project
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'models', 'plant_disease_model_final.keras')
    model = tf.keras.models.load_model(model_path)
    return model

if __name__ == "__main__":
    model = get_disease_model()
    print(model.summary())
