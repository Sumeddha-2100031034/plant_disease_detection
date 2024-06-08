import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to the dataset
DATASET_PATH = 'data/raw/PlantVillage/train'  # Update this path as needed
BATCH_SIZE = 32
IMG_SIZE = 224

def load_test_data(dataset_path, img_size, batch_size):
    datagen = ImageDataGenerator(rescale=1./255)
    test_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    return test_generator

# Load the model
model = load_model('models/plant_disease_model_final.keras')  # Ensure this path is correct

# Load test data
test_generator = load_test_data(DATASET_PATH, IMG_SIZE, BATCH_SIZE)

# Ensure the model's output layer matches the number of classes
num_classes = len(test_generator.class_indices)
if model.output_shape[-1] != num_classes:
    raise ValueError(f"Model output shape ({model.output_shape[-1]}) does not match the number of classes ({num_classes}).")

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
