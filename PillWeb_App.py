#import matplotlib.pyplot as plt
#import numpy as np
#import pandas as pd
#import seaborn as sns
#import matplotlib.image as mpimg
#import os
#from glob import glob
#from PIL import Image
#from sklearn.model_selection import train_test_split
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.metrics import confusion_matrix, classification_report
#from keras.preprocessing.image import ImageDataGenerator
#from sklearn.utils import resample
#from sklearn.svm import SVC
#from keras.utils.np_utils import to_categorical
#from sklearn.preprocessing import LabelEncoder
#from sklearn.utils.multiclass import unique_labels
#from keras.models import Sequential, load_model
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.preprocessing import image
#from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
#from keras.optimizers import Adam
#from tensorflow.keras.applications import VGG16
#import tensorflow as tf
#from tensorflow.keras import layers
#from tensorflow.keras.applications import ResNet50
#from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
#from tensorflow.keras.models import Model
#from sklearn.metrics import accuracy_score

## Load the saved model for inference
#loaded_model = load_model("saved_model.h5")

## Data Augmentation for training set
#train_datagen = ImageDataGenerator(
#    rescale=1./255,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True
#)


## Flow training images in batches using the generators
#train_generator = train_datagen.flow_from_directory(
#    directory="C:/Users/DELL USER/Desktop/Data Science/Research/Senior Design Project - Pill Dispenser/rximage/archive/SeniorProject-Pill/train",
#    target_size=(224, 224),
#    batch_size=32,
#    class_mode='categorical'
#)


## Function for image prediction
#def predict_image_class(image_path):
#    img = image.load_img(image_path, target_size = (224, 224))
#    img_array = image.img_to_array(img)
#    img_array = np.expand_dims(img_array, axis=0)
#    img_array /= 255.0  # Rescale to match the training data
#    prediction = loaded_model.predict(img_array)
#    predicted_class = np.argmax(prediction)
#    print("Number: ", predicted_class)
#    class_label = list(train_generator.class_indices.keys())[predicted_class]
#    return class_label

## Example of using the predict_image_class function
#image_path_to_predict = "test.jpg"
#predicted_class = predict_image_class(image_path_to_predict)
#print("Predicted Class:", predicted_class)




import streamlit as st
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import cv2
np_version = '1.21.0'

# Load the trained model
def load_model():
    model = tf.keras.models.load_model('saved_model.hdf5')
    return model
model = load_model()

st.set_page_config(
    page_title="Pill Dispenser Classifier App",
    page_icon="🌟",
    layout="centered",  # You can use "wide" or "centered" layout
    initial_sidebar_state="collapsed",  # Collapsed the sidebar on initial load
)
# Add your logo and resize it
logo_url = "https://images.pexels.com/photos/159211/headache-pain-pills-medication-159211.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
response = requests.get(logo_url)

if response.status_code == 200:
    # Load image from BytesIO
    logo = Image.open(BytesIO(response.content))
    logo.thumbnail((100, 100))

else:
    st.write(f"Failed to fetch the image. Status code: {response.status_code}")

# Create a layout with two columns
col1, col2 = st.columns([1, 4])

# Add the resized logo to the first column
col1.image(logo, use_column_width=True)

# Add a horizontal rule to separate the logo from the title
col1.write('<hr style="height:2px; width:530%; background-color: black">', unsafe_allow_html=True)

# Add the title to the second column
col2.title("Pill Classifier App")

# Sidebar for image upload and camera capture
uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
capture_image = st.sidebar.button("Capture Image")

def preprocess_image(image, target_size, np_version):
    img = Image.open(image).resize(target_size)
    img_array = np.asarray(img)
    
    if np_version >= '1.17.0':  # Use np_version to check the NumPy version
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
    else:
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
            
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# Check if an image is uploaded
if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Perform image classification using your model
    # Preprocess the image (resize, normalize, etc.) as needed
    img_array = preprocess_image(uploaded_image, target_size=(64, 64), np_version=np_version)

    
    # Make predictions
    predictions = model.predict(img_array)

#    # Get the class with the highest probability
    predicted_class = np.argmax(predictions)

    if (predicted_class == 0):
        st.write(f"Diagnosis: Amoxicillin 500 MG")
    elif (predicted_class == 5):
        st.write(f"Diagnosis: Apixaban 2.5 MG")
    elif (predicted_class == 6):
        st.write(f"Diagnosis: Aprepitant 80 MG")
    elif (predicted_class == 1):
        st.write(f"Diagnosis: Atomoxetine 25 MG")
    elif (predicted_class == 7):
        st.write(f"Diagnosis: Benzonatate 100 MG")
    elif (predicted_class == 2):
        st.write(f"Diagnosis: Calcitriol 0.00025 MG")
    elif (predicted_class == 8):
        st.write(f"Diagnosis: Carvedilol 3.125 MG")
    elif (predicted_class == 9):
        st.write(f"Diagnosis: Celecoxib 200 MG")
    elif (predicted_class == 10):
        st.write(f"Diagnosis: Duloxetine 30 MG")
    elif (predicted_class == 11):
        st.write(f"Diagnosis: Eltrombopag 25 MG")
    elif (predicted_class == 12):
        st.write(f"Diagnosis: Montelukast 10 MG")
    elif (predicted_class == 13):
        st.write(f"Diagnosis: Mycophenolate Mofetil 250 MG")
    elif (predicted_class == 3):
        st.write(f"Diagnosis: Oseltamivir 45 MG")
    elif (predicted_class == 14):
        st.write(f"Diagnosis: Pantoprazole 40 MG")
    elif (predicted_class == 15):
        st.write(f"Diagnosis: Pitavastatin 1 MG")
    elif (predicted_class == 16):
        st.write(f"Diagnosis: Prasugrel 10 MG")
    elif (predicted_class == 4):
        st.write(f"Diagnosis: Ramipril 5 MG")
    elif (predicted_class == 17):
        st.write(f"Diagnosis: Saxagliptin 5 MG")
    elif (predicted_class == 18):
        st.write(f"Diagnosis: Sitagliptin 50 MG")
    elif (predicted_class == 19):
        st.write(f"Diagnosis: Tadalafil 5 MG")


# Add a footer with additional information
st.text("")
st.text("")
st.text("Pills include: Amoxicillin 500 MG, Apixaban 2.5 MG, Aprepitant 80 MG, Atomoxetine 25 MG, Benzonatate 100 MG, Calcitriol 0.00025 MG, Carvedilol 3.125 MG, Celecoxib 200 MG, Duloxetine 30 MG, Eltrombopag 25 MG, Montelukast 10 MG, Mycophenolate Mofetil 250 MG, Oseltamivir 45 MG, Pantoprazole 40 MG, Pitavastatin 1 MG, Prasugrel 10 MG, Ramipril 5 MG, Saxagliptin 5 MG, Sitagliptin 50 MG, Tadalafil 5 MG")
st.text("This is a simple pill classifier web app.")
