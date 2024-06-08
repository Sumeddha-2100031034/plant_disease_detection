import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to the PlantVillage dataset
DATASET_PATH = r'data/raw/PlantVillage/train'  # Update this path as needed
IMG_SIZE = 224
BATCH_SIZE = 32

def load_images_and_labels(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The specified dataset path does not exist: {dataset_path}")
    
    images = []
    labels = []
    classes = os.listdir(dataset_path)
    class_dict = {cls: idx for idx, cls in enumerate(classes)}
    
    for cls in classes:
        class_path = os.path.join(dataset_path, cls)
        if not os.path.isdir(class_path):
            print(f"Warning: Skipping non-directory item: {class_path}")
            continue
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            
            # Check if the image was successfully loaded
            if img is None:
                print(f"Warning: Failed to load image {img_path}")
                continue
            
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(class_dict[cls])
    
    if not images or not labels:
        raise ValueError("No images or labels loaded. Check dataset path and file integrity.")
    
    return np.array(images), np.array(labels), class_dict

def preprocess_data(images, labels):
    images = images / 255.0  # Normalize images
    labels = np.array(labels)
    return train_test_split(images, labels, test_size=0.2, random_state=42)

def create_data_generators(x_train, x_val, y_train, y_val):
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
    val_generator = val_datagen.flow(x_val, y_val, batch_size=BATCH_SIZE)
    
    return train_generator, val_generator

if __name__ == "__main__":
    images, labels, class_dict = load_images_and_labels(DATASET_PATH)
    x_train, x_val, y_train, y_val = preprocess_data(images, labels)
    train_generator, val_generator = create_data_generators(x_train, x_val, y_train, y_val)
    
    # Save preprocessed data
    np.save('x_train.npy', x_train)
    np.save('x_val.npy', x_val)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('class_dict.npy', class_dict)
