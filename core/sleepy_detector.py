import cv2
import dlib
import numpy as np
from functools import lru_cache

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def detect_eye_aspect_ratio(eye_points):
    A = euclidean_distance(eye_points[1], eye_points[5])
    B = euclidean_distance(eye_points[2], eye_points[4])
    C = euclidean_distance(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

@lru_cache(maxsize=1)
def get_predictor():
    return dlib.shape_predictor("Models/shape_predictor_68_face_landmarks.dat")

def check_sleepy(frame, ear_threshold: float = 0.25):
    if frame is None or frame.size == 0:
        return True  # Invalid frame, assume sleepy

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = get_predictor()
    faces = face_detector(gray)

    if len(faces) == 0:
        return True  # No face detected = possibly not paying attention

    for face in faces:
        landmarks = shape_predictor(gray, face)
        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

        if len(left_eye) != 6 or len(right_eye) != 6:
            return True  # Eye landmarks not detected properly

        EAR = (detect_eye_aspect_ratio(left_eye) + detect_eye_aspect_ratio(right_eye)) / 2.0
        if EAR < ear_threshold:
            return True  # Eyes appear to be closed

    return False  # Eyes are detected and open

