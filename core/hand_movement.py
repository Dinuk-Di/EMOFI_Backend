# import cv2
# import numpy as np
# from ultralytics import YOLO
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# HAND_YOLO = YOLO("Models/yolov8n.pt")
# HAND_MODEL = load_model("Models/hand_movement.h5")
# HAND_CLASSES = ['Boring', 'Neutral', 'Stress']

# def detect_hand(frame, yolo_conf: float = 0.3, conf_thres: float = 0.5, min_person_area: int = 10000) -> list:
#     if frame is None or frame.size == 0:
#         return []

#     frame = cv2.flip(frame, 1)
#     results = HAND_YOLO(frame, classes=[0], conf=yolo_conf, verbose=False)

#     bbox = None
#     largest_area = 0

#     for r in results:
#         if r.boxes is not None:
#             for box in r.boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].tolist()
#                 area = (x2 - x1) * (y2 - y1)
#                 if area > largest_area and area >= min_person_area:
#                     largest_area = area
#                     bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

#     if bbox:
#         x, y, w, h = bbox
#         cropped = frame[y:y+h, x:x+w]
#         if cropped.size > 0:
#             resized = cv2.resize(cropped, (224, 224))
#             rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
#             preprocessed = preprocess_input(rgb_img)
#             input_tensor = np.expand_dims(preprocessed, axis=0)

#             preds = HAND_MODEL.predict(input_tensor, verbose=0)[0]
#             idx = int(np.argmax(preds))
#             conf = float(np.max(preds))

#             if conf >= conf_thres and 0 <= idx < len(HAND_CLASSES):
#                 return [HAND_CLASSES[idx]]

#     return []


import cv2
from ultralytics import YOLO
from utils.tools import resource_path

HAND_MODEL = YOLO(resource_path('Models/hand_best.pt'))  # update path as needed
HAND_CLASSES = ['Boring', 'Neutral', 'Stress']

def detect_hand(frame, conf_thres: float = 0.5):
    """
    Classifies the hand movement in the frame using YOLOv8 classification model.
    Returns the predicted class label if confidence is above threshold.
    """
    if frame is None or frame.size == 0:
        return []

    # YOLOv8-cls expects RGB image
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run classification (direct call, not .predict)
    results = HAND_MODEL(rgb_img)
    probs = getattr(results[0], "probs", None)

    # Use YOLOv8 classification result attributes for top1/top1conf
    if probs and float(probs.top1conf) > conf_thres:
        idx = int(probs.top1)
        if 0 <= idx < len(HAND_CLASSES):
            return [HAND_CLASSES[idx]]

    return []
