!pip install flask-ngrok
!pip install flask

!pip install pyngrok
!pip install streamlit opencv-python-headless tensorflow pyngrok

!pip install opencv-python-headless

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import streamlit as st
from pyngrok import ngrok, conf
import os

# Close any existing tunnels
for tunnel in ngrok.get_tunnels():
    ngrok.disconnect(tunnel.public_url)

# Set up ngrok
ngrok.set_auth_token("2jZMbmksjl96RvHqjlPkPABNopD_5NBfiUFNChZMRe3S48Z71")  # Replace with your actual ngrok authtoken

# Configure ngrok
config = conf.PyngrokConfig(auth_token="2jZMbmksjl96RvHqjlPkPABNopD_5NBfiUFNChZMRe3S48Z71")
public_url = ngrok.connect(5000, "http", pyngrok_config=config)
print(f" * ngrok tunnel URL: {public_url}")

app = Flask(__name__)

# Load pre-trained Inception V3 model
model = InceptionV3(weights='imagenet')

UPLOAD_FOLDER = '/content/uploads'
FRAMES_FOLDER = '/content/frames'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

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

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            if file.content_length > MAX_CONTENT_LENGTH:
                return jsonify({'error': 'File size exceeds limit'}), 413
            filename = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filename)
            return process_video(filename)
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    return render_template_string('''
        <!doctype html>
        <title>Upload Video and Search Objects</title>
        <h1>Upload a video and search for objects</h1>
        <form method=post enctype=multipart/form-data>
            <input type=file name=file accept=".mp4,.avi,.mov">
            <input type=text name=search placeholder="Enter object to search">
            <input type=submit value=Upload>
        </form>
    ''')

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    all_objects = {}
    search_query = request.form.get('search', '').lower()
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

    if search_query:
        if search_query in all_objects:
            return jsonify({
                'message': f"Object '{search_query}' found in frames: {all_objects[search_query]}",
                'frames': [f"frame_{fc}.jpg" for fc in all_objects[search_query]]
            })
        else:
            return jsonify({'error': "Object doesn't exist!!!"}), 404
    
    return jsonify({'detected_objects': all_objects})

if __name__ == '__main__':
    app.run(port=5000)
