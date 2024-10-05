# app.py
from flask import Flask, Response, render_template, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import queue
#----------- DEPLOY THINGS --------------
import os

app = Flask(__name__)

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_queue = queue.Queue()

def text_to_speech_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        tts_engine.say(text)
        tts_engine.runAndWait()
        tts_queue.task_done()

# Start TTS worker thread
tts_thread = threading.Thread(target=text_to_speech_worker, daemon=True)
tts_thread.start()

# Load the model
model = tf.keras.models.load_model('Sign_classifier.h5')

# Class labels
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                'del', 'nothing', 'space']

def preprocess_image(roi, target_size=(224, 224)):
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, target_size)
    roi = tf.keras.applications.mobilenet.preprocess_input(roi)
    roi = np.expand_dims(roi, axis=0)
    return roi

# Global variables for video capture
camera = None
box_top_left = (100, 100)
box_bottom_right = (350, 350)

#----------- LOCAL THINGS --------------
# def generate_frames():
#     global camera
#     if camera is None:
#         camera = cv2.VideoCapture(0)

#     while True:
#         success, frame = camera.read()
#         if not success:
#             break

#         cv2.rectangle(frame, box_top_left, box_bottom_right, (0, 255, 0), 2)
        
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
        
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
#----------- DEPLOY THINGS --------------
def generate_frames():
    global camera
    if camera is None:
        try:
            camera = cv2.VideoCapture(0)
        except:
            # Return a blank frame if camera is not available
            blank_frame = np.zeros((480, 640, 3), np.uint8)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            return

    while True:
        try:
            success, frame = camera.read()
            if not success:
                break
            
            cv2.rectangle(frame, box_top_left, box_bottom_right, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except:
            break
#------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    global camera
    if camera is None:
        return jsonify({'error': 'Camera not initialized'})

    success, frame = camera.read()
    if not success:
        return jsonify({'error': 'Failed to capture frame'})

    roi = frame[box_top_left[1]:box_bottom_right[1], 
                box_top_left[0]:box_bottom_right[0]]
    processed_frame = preprocess_image(roi)
    predictions = model.predict(processed_frame)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class_index]

    return jsonify({'prediction': predicted_label})

@app.route('/speak_text', methods=['POST'])
def speak_text():
    data = request.json
    text = data.get('text', '')
    if text:
        tts_queue.put(text)
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'No text provided'})


#----------- LOCAL THINGS --------------
# if __name__ == '__main__':
#     app.run(debug=True)

#----------- DEPLOY THINGS --------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)