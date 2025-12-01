import logging
import numpy as np
from ultralytics import YOLO
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def _load_yolo(model_path: str = "Models/yolov8n.pt"):
    return YOLO(model_path)

def human_present(frame, conf_thres: float = 0.4, imgsz: int = 320) -> bool:
    if frame is None or frame.size == 0:
        return False

    model = _load_yolo()
    results = model(frame, classes=[0], conf=conf_thres, imgsz=imgsz, verbose=False)
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            return True
    return False



