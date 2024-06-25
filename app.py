import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os
from mtcnn import MTCNN
import numpy as np
import keras 
from keras.applications import ResNet50

st.title('Which Bollywood celebrity are you?')

uploaded_image = None

col1, col2 = st.columns(2)
with col2:
    uploaded_image_1 = st.file_uploader("Choose an image ", accept_multiple_files=False)

with col1:
    uploaded_image_2 = st.camera_input("Take a picture")

try:
    if uploaded_image_1 is not None:
        uploaded_image = uploaded_image_1

    if uploaded_image_2 is not None:
        uploaded_image = uploaded_image_2
except:
    pass

detector = MTCNN()

model = ResNet50(include_top=False, input_shape=(224, 224, 3), pooling='avg')

feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filename.pkl', 'rb'))

def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

def extract_features(img_path, model, detector):
    img = Image.open(img_path)
    img = img.convert("RGB")
    
    img = np.array(img)  # Convert PIL Image to numpy array
    
    results = detector.detect_faces(img)

    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]

    # Resize and preprocess the face image
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)
    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = keras.applications.resnet.preprocess_input(expanded_img)  # Using ResNet preprocess_input

    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(feature_list, features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

if uploaded_image is not None:
    # Save the image in a directory
    if save_uploaded_image(uploaded_image):
        # Load the image
        display_image = Image.open(uploaded_image)

        # Extract the features
        features = extract_features(os.path.join('uploads', uploaded_image.name), model, detector)
        
        # Recommend
        index_pos = recommend(feature_list, features)
        predicted_actor = " ".join(filenames[index_pos].split('\\')[4].split('.')[0].split('_'))
        
        # Display
        col1, col2 = st.columns(2)

        with col1:
            st.header('Your uploaded image')
            st.image(display_image)
        
        with col2:
            st.header("Seems like " + predicted_actor)
            st.image(filenames[index_pos], width=300)
