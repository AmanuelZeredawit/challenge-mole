

import pandas as pd
import streamlit as st
import sys
import warnings
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import os
from fastai.vision.all import *
from fastai.metrics import *
from pathlib import Path
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.insert(0, "D:\Arai4_Projects\challenge-mole\preprocessing")
sys.path.insert(0, "D:\Arai4_Projects\challenge-mole\modeling")
from get_path import get_filenames
from get_path import group_filenames
import pickle

def main():

    global model
    # Load the saved .h5 file using joblib
    model_file = joblib.load("model.joblib")
    # Load the model from the .h5 file
    model = tf.keras.models.load_model(model_file)

    st.title("🧑🏽‍⚕️ SKINCARE 👩🏻‍⚕️")
    menu = ['Home','classifier']
    choice = st.sidebar.selectbox("Menu", menu)


    def is_malignant(test_image):
        test_image = Image.open(test_image).resize((256,256))
        x = image.img_to_array(test_image)
        x = np.expand_dims(x, axis=0)
        datagen = ImageDataGenerator(rescale=1./255)
        x = datagen.standardize(x)
        preds = model.predict(x)
        result = np.argmax(preds, axis=1)[0]
        return  result



    if choice == "Home":
        im =Image.open("../images/logo.jpg")
        st.image(im, width=1000)

    elif choice == "classifier":
        map_result = {0:'Benign', 1: 'Malignant'}
        

        input = st.sidebar.selectbox("Are you going to import or scan image?", ("Import","Scan"))
        if input== "Import":
            
            loaded_image = st.sidebar.file_uploader("Upload the test image", type=["jpg"])
           
            if loaded_image:
                 st.image(loaded_image)
                 if st.button("Test",key="test_1"):
                    result = is_malignant(loaded_image)
                    st.write("Result is :  ", map_result[result])

        elif input == "Scan":
            picture = st.camera_input("Take a picture")
            if picture:
                if st.button("Test",key= 'test_2'):
                    result = is_malignant(picture)
                    st.write('Result is : ', map_result[result])

        
        



if __name__ == '__main__':
    main()