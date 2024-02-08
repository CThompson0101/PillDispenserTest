import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.image as mpimg
import os
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import resample
from sklearn.svm import SVC
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels
from keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score


# Define paths to your train and valid directories

train_dir = "train"
valid_dir = "valid"


# Print sample image
image_location = "train/tadalafil 5 MG/tadalafil 5 MG (11).jpg"
img = mpimg.imread(image_location)
plt.imshow(img)
plt.axis('off')
plt.show()


# Data Augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescaling for validation set
valid_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches using the generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(train_generator.class_indices), activation='softmax'))


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,  # Adjust as needed
    validation_data=valid_generator
)

# Evaluate the model
test_loss, test_acc = model.evaluate(valid_generator)
print("Test Accuracy:", test_acc)

# Save the model
model.save("saved_model.hdf5")

# Load the saved model for inference
#loaded_model = load_model("saved_model.hdf5")


# Function for image prediction
#def predict_image_class(image_path):
#    img = image.load_img(image_path, target_size = (64, 64))
#    img_array = image.img_to_array(img)
#    img_array = np.expand_dims(img_array, axis=0)
#    img_array /= 255.0  # Rescale to match the training data
#    prediction = loaded_model.predict(img_array)
#    predicted_class = np.argmax(prediction)
#    class_label = list(train_generator.class_indices.keys())[predicted_class]
#    return class_label

## Example of using the predict_image_class function
#image_path_to_predict = "test.jpg"
#predicted_class = predict_image_class(image_path_to_predict)
#print("Predicted Class:", predicted_class)
