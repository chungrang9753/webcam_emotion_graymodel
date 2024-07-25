import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 감정 레이블
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 모델 로드
try:
    model = load_model('emotion_model.h5')
except Exception as e:
    raise IOError(f'Could not load model: {e}')

# 얼굴 탐지기 로드 (OpenCV 제공)
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    raise IOError(f'Could not load haarcascade file from {face_cascade_path}')

# 웹캠 캡처 시작
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError('Could not open webcam')

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 얼굴 탐지를 위한 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 탐지
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    print(f"Number of faces detected: {len(faces)}")  # 디버깅을 위한 얼굴 탐지 개수 출력

    for (x, y, w, h) in faces:
        # 얼굴 영역 추출
        face = gray[y:y+h, x:x+w]  # 그레이스케일 이미지에서 얼굴 추출

        # 모델 입력 크기에 맞게 조정
        face_resized = cv2.resize(face, (96, 96))
        face_resized = face_resized / 255.0
        face_resized = face_resized.reshape(1, 96, 96, 1)  # 흑백 이미지를 위해 마지막 차원을 1로 설정

        # 감정 예측
        try:
            prediction = model.predict(face_resized)
            emotion = emotion_labels[np.argmax(prediction)]
        except Exception as e:
            print(f'Error in prediction: {e}')
            emotion = 'unknown'

        # 얼굴 주위에 네모 그리기 및 감정 레이블 표시
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # 결과 보여주기
    cv2.imshow('Emotion Recognition', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 캡처 종료 및 창 닫기
cap.release()
cv2.destroyAllWindows()
