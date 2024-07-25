from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
import cv2
from keras.models import load_model
import time
import threading
from collections import Counter
import csv

app = Flask(__name__)
socketio = SocketIO(app)


model_path = 'emotion_model.h5'
model = load_model(model_path)

labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
img_size = 96

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

is_detecting = False
emotions_list = []
emotion_thread = None

# CSV 파일에 헤더 추가
with open('emotion_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['timestamp', 'emotion'])

def get_emotion():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "No Face Detected"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return "No Face Detected"

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (img_size, img_size))
        face = face.astype('float32') / 255.0
        face = np.reshape(face, (1, img_size, img_size, 1))
        
        prediction = model.predict(face)
        emotion = labels[np.argmax(prediction)]
        return emotion

    return "No Face Detected"

def detect_emotions():
    global is_detecting, emotions_list
    while is_detecting:
        emotion = get_emotion()
        if emotion != "No Face Detected":
            emotions_list.append(emotion)
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            with open('emotion_data.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, emotion])
        socketio.sleep(3)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/measurement')
def measurement():
    return render_template('measurement.html')

@app.route('/statistics')
def statistics():
    return render_template('statistics.html')

@app.route('/consulting')
def consulting():
    return render_template('consulting.html')

@socketio.on('start_detection')
def start_detection():
    global is_detecting, emotion_thread, emotions_list
    if not is_detecting:
        is_detecting = True
        emotions_list = []
        emotion_thread = threading.Thread(target=detect_emotions)
        emotion_thread.start()
        emit('status', {'message': 'Emotion detection started'})

@socketio.on('stop_detection')
def stop_detection():
    global is_detecting, emotion_thread, emotions_list
    is_detecting = False
    if emotion_thread:
        emotion_thread.join()
    emotion_counts = dict(Counter(emotions_list))
    emit('emotions', {'emotions': emotions_list, 'counts': emotion_counts})

@app.route('/get_statistics', methods=['GET'])
def get_statistics():
    emotion_counts = Counter()
    
    # CSV 파일에서 데이터를 읽어와 감정별로 카운트를 계산
    with open('emotion_data.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 헤더 건너뛰기
        for row in reader:
            emotion = row[1]
            emotion_counts[emotion] += 1
    
    # 가장 많이 나온 감정 찾기
    max_emotion = max(emotion_counts, key=emotion_counts.get)
    
    return jsonify({
        'emotion_counts': dict(emotion_counts),
        'max_emotion': max_emotion
    })

if __name__ == '__main__':
    socketio.run(app, debug=True)
