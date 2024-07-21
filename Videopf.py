import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import streamlit as st
from pyngrok import ngrok, conf
import os

# Set up ngrok
ngrok.set_auth_token("2jYqJSc13BE5i1jcCGdpFnTpnPy_7KTJqTarwcDwXDbLWfaoa") 
# Configure ngrok
config = conf.PyngrokConfig(auth_token="2jYqJSc13BE5i1jcCGdpFnTpnPy_7KTJqTarwcDwXDbLWfaoa")
ngrok_tunnel = ngrok.connect(8501, "http", pyngrok_config=config)
public_url = ngrok_tunnel.public_url
st.write(f"ngrok tunnel URL: {public_url}")

# Load pre-trained Inception V3 model
model = InceptionV3(weights='imagenet')

UPLOAD_FOLDER = 'uploads'
FRAMES_FOLDER = 'frames'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_frame(frame):
    img = cv2.resize(frame, (299, 299))  # Inception V3 expects 299x299 images
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def detect_objects(frame):
    processed_frame = process_frame(frame)
    predictions = model.predict(processed_frame)
    results = decode_predictions(predictions, top=5)[0]
    return [result[1].lower() for result in results]

def process_video(video_path, search_query):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    all_objects = {}
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1

    cap.release()

    # Process every 30th frame
    for i, frame in enumerate(frames[::30]):
        frame_number = i * 30
        objects = detect_objects(frame)
        for obj in objects:
            if obj not in all_objects:
                all_objects[obj] = []
            all_objects[obj].append(frame_number)

        if search_query and search_query in objects:
            frame_filename = f"frame_{frame_number}.jpg"
            frame_path = os.path.join(FRAMES_FOLDER, frame_filename)
            cv2.imwrite(frame_path, frame)

    return all_objects

# Streamlit app
st.title('Upload Video and Search Objects')
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
search_query = st.text_input("Enter object to search")

if uploaded_file is not None:
    if allowed_file(uploaded_file.name):
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write("Processing video...")
        all_objects = process_video(file_path, search_query.lower())
        
        if search_query:
            if search_query.lower() in all_objects:
                st.write(f"Object '{search_query}' found in frames: {all_objects[search_query.lower()]}")
                for frame_num in all_objects[search_query.lower()]:
                    frame_img_path = os.path.join(FRAMES_FOLDER, f"frame_{frame_num}.jpg")
                    st.image(frame_img_path, caption=f"Frame {frame_num}")
            else:
                st.write("Object doesn't exist!!!")
        else:
            st.write("Detected objects in the video:")
            st.json(all_objects)
    else:
        st.write("Invalid file type")
