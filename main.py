# Modules
import streamlit as st
from pyrebase import pyrebase
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.utils import load_img, img_to_array
from keras import preprocessing
from streamlit_option_menu import option_menu
import cv2
from keras.models import load_model
from streamlit import session_state as state
import time
from streamlit_extras.stateful_button import button
import os
import tempfile

# To store Configuration key in database
firebaseconfig = {
    'apiKey': "AIzaSyBjwkNm-FLWgH5Rge6dxby_dB5ySKNG6yk",
    'authDomain': "applediseasedetection1.firebaseapp.com",
    'databaseURL': "https://applediseasedetection1-default-rtdb.asia-southeast1.firebasedatabase.app",
    'projectId': "applediseasedetection1",
    'storageBucket': "applediseasedetection1.appspot.com",
    'messagingSenderId': "229891354768",
    'appId': "1:229891354768:web:5c1e0d28931d250f10e815",
    'measurementId': "G-PBVPDY303C"
}


# Firebase Authentication
firebase = pyrebase.initialize_app(firebaseconfig)
auth = firebase.auth()

# Database
db = firebase.database()
storage = firebase.storage()
# Authentication
with st.sidebar:
    choice = option_menu(
        menu_title='Main Menu',
        options=['Login', 'Sign Up'],
        icons=['door-open', 'app-indicator'],
        orientation='horizontal'
    )

st.title('Apple Leaf Disease Detection')
st.sidebar.title('User Authentication')
st.image('welcome.png')

# User Input

email = st.sidebar.text_input('Please enter your email address')
password = st.sidebar.text_input('Please enter your password', type='password')

# Sign Up Block


def Auth():
    user = auth.sign_in_with_email_and_password(email, password)
    return user["registered"]


if choice == 'Sign Up':
    handle = st.sidebar.text_input('Please enter your Username')
    submit = st.sidebar.button('Create My Account')

    if submit:
        user = auth.create_user_with_email_and_password(email, password)
        st.success('Your account has been created successfully')
        st.balloons()
        # Sign in
        user = auth.sign_in_with_email_and_password(email, password)
        db.child(user['localId']).child("Name").set(handle)
        db.child(user['localId']).child("ID").set(user['localId'])
        st.title('Welcome ' + handle)
        st.info('Login via dropdown Login Option')


# Login Block
if choice == 'Login':
    button_clicked = st.session_state.get('login_button_clicked', False)
    if button_clicked:
        if Auth():
            st.success('Logged in successfully')
            st.title('Welcome')
        else:
            st.error('Incorrect email or password')

    login = st.sidebar.checkbox('Login')

    if login:
        user = auth.sign_in_with_email_and_password(email, password)

        st.write(
            '<style>div.row-widget.stRadio > div{flex-direction:column;}</style>', unsafe_allow_html=True)

        bio = st.radio('Pages', ['Home', 'Detection'])

        # Home Page
        if bio == 'Home':
            st.header('Apple Leaf Disease Detection')
            st.image("welcome2.png")
            st.write('''Welcome to the Apple Leaf Disease Detection system - the fast and easy way to identify and manage diseases in your apple leaves. This project named "Apple Leaf Disease Detection using CNN" is a minor project from BEI076.
    The model is trained on a large dataset of annotated apple leaf images, and is capable of detecting several common diseases such as apple scab, cedar apple rust, and black rot.''')

        # Detection Page

        elif bio == 'Detection':
            def is_leaf(uploaded_file):
                # Save uploaded file to a temporary directory
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(uploaded_file.read())
                    img_path = temp_file.name

                img = cv2.imread(img_path)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                lower_green = np.array([25, 52, 72])
                upper_green = np.array([102, 255, 255])
                mask = cv2.inRange(hsv, lower_green, upper_green)
                ratio = np.sum(mask == 255) / (mask.shape[0] * mask.shape[1])
                os.unlink(temp_file.name)  # delete the temporary file
                if ratio > 0.05:  # threshold for detecting leaf in the image
                    return True
                else:
                    return False
            st.title('Detect Image')
            plant_image = st.file_uploader(
                "Choose an image to predict...", type=['jpg', 'png', 'jpeg', 'JPG'])

            if st.checkbox("Predict"):
                if 'predict_clicked' not in st.session_state:
                    st.session_state.predict_clicked = False

                st.session_state.predict_clicked = True
                if plant_image is not None:
                    if not is_leaf(plant_image):
                        st.warning("The uploaded image is not a leaf image.")
                    else:
                        st.session_state.treatment_clicked = False
                        model = load_model(
                            r"C:\Users\gauta\Desktop\FINAL REPORT\balanced_cnn_model.h5")
                        # Progress bar
                        progress_text = "Operation in progress. Please wait."
                        my_bar = st.progress(0, text=progress_text)
                        for percent_complete in range(100):
                            time.sleep(0.1)
                            my_bar.progress(percent_complete +
                                            1, text=progress_text)

                        # image preprocessing
                        CLASS_NAMES = ['apple_scab', 'black_rot',
                                       'cedar_apple_rust', 'healthy']

                        new_image = load_img(
                            plant_image, target_size=(256, 256))
                        new_image_array = img_to_array(new_image)
                        new_image_array = np.expand_dims(
                            new_image_array, axis=0)
                        new_image_array = new_image_array / 255.0

                        predictions = model.predict(new_image_array)
                        class_index = np.argmax(predictions)
                        class_label = CLASS_NAMES[class_index]
                        class_prob = predictions[0][class_index]

                        st.image(new_image_array,
                                 caption=class_label, channels="BGR")
                        st.title("The uploaded image is a {} with a probability of {:.2f}%".format(
                            class_label, class_prob * 100))

                   
                   

                    # if st.session_state.treatment_clicked:
                    if st.button("Treatment"):
                        if 'treatment_clicked' not in st.session_state:
                            st.session_state.treatment_clicked = False
                        st.session_state.treatment_clicked = True
                        st.title('Apple Leaf Disease Treatment')
                        if class_label  == 'apple_scab':
                            st.write(
                                'The detected disease is apple scab. Treatment includes using fungicides and improving air circulation around the trees through pruning and thinning.')
                        elif class_label  == 'black_rot':
                            st.write(
                                'The detected disease is black rot. Treatment includes using fungicides and pruning to improve air circulation and light penetration in the canopy.')
                        elif class_label  == 'cedar_apple_rust':
                            st.write(
                                'The detected disease is cedar apple rust. Treatment includes using fungicides, pruning infected branches, and removing nearby cedar trees if they are heavily infected.')
                        elif class_label  == 'healthy':
                            st.write(
                                'The leaf is healthy. No need for treatment')
                            
                        else:
                            st.write('Please upload a leaf image.')

                   
 