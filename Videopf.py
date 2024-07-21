import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import streamlit as st
import os
import tempfile

# Load pre-trained Inception V3 model
@st.cache_resource
def load_model():
    return InceptionV3(weights='imagenet')

model = load_model()

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB limit

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
            st.image(frame, caption=f"Frame {frame_number}: {search_query} detected")

    if search_query:
        if search_query in all_objects:
            st.success(f"Object '{search_query}' found in frames: {all_objects[search_query]}")
        else:
            st.error("Object doesn't exist!!!")
    
    return all_objects

def main():
    st.title("Video Object Detection")

    uploaded_file = st.file_uploader("Choose a video file", type=ALLOWED_EXTENSIONS)
    search_query = st.text_input("Enter object to search (optional)").lower()

    if uploaded_file is not None:
        if uploaded_file.size > MAX_CONTENT_LENGTH:
            st.error("File size exceeds limit (50 MB)")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            st.video(tmp_file_path)

            if st.button("Process Video"):
                with st.spinner("Processing video..."):
                    all_objects = process_video(tmp_file_path, search_query)
                
                if not search_query:
                    st.json(all_objects)

            os.unlink(tmp_file_path)

if __name__ == "__main__":
    main()
