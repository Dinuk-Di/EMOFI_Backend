import cv2
from ultralytics import YOLO

EMOTION_MODEL = YOLO("Models/best.pt")
EMOTION_CLASS_NAMES = ['Angry', 'Boring', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Stress', 'Suprise']

def get_emotion(frame, conf_thres: float = 0.5) -> list:
    if frame is None or frame.size == 0:
        return []

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        results = EMOTION_MODEL(face_img, verbose=False)
        probs = getattr(results[0], "probs", None)

        if probs and probs.top1conf > conf_thres:
            idx = probs.top1
            if 0 <= idx < len(EMOTION_CLASS_NAMES):
                return [EMOTION_CLASS_NAMES[idx]]

    return []
