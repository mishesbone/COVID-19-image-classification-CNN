#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 03:07:39 2023

@author: roboteknologies
"""
import matplotlib.image as mpimg
import os
import matplotlib.pyplot as plt
import random
import shutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Set the path to your dataset directory
dataset_dir = '/home/roboteknologies/Downloads/COVID'

# List of class names (assuming your classes are the subdirectories)
class_names = os.listdir(dataset_dir)

# Visualize a few random images from each class
num_samples_per_class = 3

for class_name in class_names:
    class_dir = os.path.join(dataset_dir, class_name)
    image_files = [f for f in os.listdir(class_dir) if f.endswith('.png') or f.endswith('.jpg')]  # Update file extensions
    
    # Add print statements here to help with debugging
    print(f"Class Name: {class_name}")
    print(f"Class Directory: {class_dir}")
    print(f"Number of Image Files: {len(image_files)}")
    
    random.shuffle(image_files)
    
    # Display a few random images from this class
    for i in range(min(num_samples_per_class, len(image_files))):
        sample_image_name = random.choice(image_files)
        sample_image_path = os.path.join(class_dir, sample_image_name)
        
        # Load and display the image
        img = mpimg.imread(sample_image_path)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')
        plt.show()

# Define the split ratios
train_ratio = 0.6
validation_ratio = 0.1
test_ratio = 0.3

# Create directories for training, validation, and test sets if they don't exist
train_dir = 'train_data'
validation_dir = 'validation_data'
test_dir = 'test_data'

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(validation_dir):
    os.makedirs(validation_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Create dictionaries to store counts of samples per class in each set
samples_per_class = {class_name: {'train': 0, 'validation': 0, 'test': 0} for class_name in class_names}

# Iterate through each class folder (moderate, severe, mild)
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    
    # Get a list of all the image files in the class folder
    image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
    
    # Shuffle the list of image files
    random.shuffle(image_files)
    
    # Calculate the number of images for each split
    num_images = len(image_files) # Calculate the number of all images
    num_train = int(num_images * train_ratio) # Calculate the number of train images
    num_validation = int(num_images * validation_ratio) # Calculate the number of validation images
    num_test = num_images - num_train - num_validation  # Calculate the number of test images
    
    # Ensure equal class representation in each set
    samples_per_class[class_name]['train'] = num_train
    samples_per_class[class_name]['validation'] = num_validation
    samples_per_class[class_name]['test'] = num_test
    
    # Split the images into train, validation, and test sets
    train_images = image_files[:num_train]
    validation_images = image_files[num_train:num_train + num_validation]
    test_images = image_files[num_train + num_validation:]
    
    # Move the images to their respective directories for this class
    for image in train_images:
        src_path = os.path.join(class_dir, image)
        dest_path = os.path.join(train_dir, class_name, image)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(src_path, dest_path)
    
    for image in validation_images:
        src_path = os.path.join(class_dir, image)
        dest_path = os.path.join(validation_dir, class_name, image)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(src_path, dest_path)
    
    for image in test_images:
        src_path = os.path.join(class_dir, image)
        dest_path = os.path.join(test_dir, class_name, image)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(src_path, dest_path)

print("Dataset split into train, validation, and test sets successfully with equal class distribution.")

# Define data directories for training, validation, and testing
train_data_dir = '/home/roboteknologies/python projects/tarinis/train_data'
validation_data_dir = '/home/roboteknologies/python projects/tarinis/validation_data'
test_data_dir = '/home/roboteknologies/python projects/tarinis/test_data'


# Set up data generators with image augmentation for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Specify batch size and target image size
batch_size = 32
image_size = (224, 224)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for multiclass classification
    save_format='png'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for multiclass classification
    save_format='png'
)

# Define the CNN model

# Model Architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))  # Add another convolutional layer
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))  # Reduce dropout rate
model.add(Dense(len(class_names), activation='softmax'))  # Use the number of classes as output neurons

# Define a learning rate scheduler
def step_decay(epoch):
    initial_lr = 0.001  # Initial learning rate
    drop = 0.5  # Factor by which the learning rate will be reduced
    epochs_drop = 10  # Number of epochs after which to reduce the learning rate
    lr = initial_lr * (drop ** (epoch // epochs_drop))
    return lr

lr_scheduler = LearningRateScheduler(step_decay)

# Define hyperparameters
learning_rate = 0.001
epochs = 1  # You can adjust this number based on validation curve analysis

# Create callbacks for monitoring and adjusting hyperparameters
early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor for early stopping
    patience=10,         # Number of epochs with no improvement to wait
    restore_best_weights=True  # Restore model to best weights when stopped
)

model_checkpoint = ModelCheckpoint(
    'best_model.h5',  # File to save the best model weights
    monitor='val_loss',  # Metric to monitor for saving the best model
    save_best_only=True,  # Save only the best model
    save_weights_only=True  # Save only the model weights (no architecture)
)

# Create the optimizer with the specified learning rate
optimizer = Adam(learning_rate=learning_rate)

# Compile the model with the optimizer
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with the defined batch size and epochs, using callbacks
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    batch_size=batch_size,
    callbacks=[early_stopping, model_checkpoint, lr_scheduler]
)

# Set up data generator for testing
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for multiclass classification
    save_format='png'
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# 1. Accuracy and Loss Plots
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# 2. Confusion Matrix
# Generate predictions for the test set
test_predictions = model.predict(test_generator)
test_predictions = [class_names[i] for i in test_predictions.argmax(axis=1)]

# Get the true labels
true_labels = [class_names[i] for i in test_generator.classes]

# Create a confusion matrix
confusion = confusion_matrix(true_labels, test_predictions)

# Plot the confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# 3. Classification Report
report = classification_report(true_labels, test_predictions, target_names=class_names)
print("Classification Report:\n", report)

from sklearn.ensemble import BaggingClassifier

# Create a bagging classifier for your CNN model
bagging_model = BaggingClassifier(base_estimator=model, n_estimators=10, random_state=0)

# Fit the bagging model on your training data
bagging_model.fit(train_generator, train_generator.classes)

# Evaluate the bagging model on the test set
bagging_accuracy = bagging_model.score(test_generator, test_generator.classes)
print(f"Bagging Model Test Accuracy: {bagging_accuracy}")

from sklearn.ensemble import AdaBoostClassifier

# Create an AdaBoost classifier for your CNN model
ada_boost_model = AdaBoostClassifier(base_estimator=model, n_estimators=50, random_state=0)

# Fit the AdaBoost model on your training data
ada_boost_model.fit(train_generator, train_generator.classes)

# Evaluate the AdaBoost model on the test set
ada_boost_accuracy = ada_boost_model.score(test_generator, test_generator.classes)
print(f"AdaBoost Model Test Accuracy: {ada_boost_accuracy}")
