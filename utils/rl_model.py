import os
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, List
from sklearn.linear_model import Ridge

from .models import AppRecommendation

CONTEXT_DIM = 12
ACTION_DIM = 8   
N_FEATURES = CONTEXT_DIM + ACTION_DIM

# --- RL Model Class ---

class MockLinUCB:
    """
    This class acts as the interface for our RL model.
    It holds a trained Ridge regression model to predict scores (rewards).
    """
    def __init__(self, n_features):
        self.n_features = n_features
        self.model_ready = False
        self.model: Ridge = None
        print(f"MockLinUCB initialized with {n_features} features.")

    def predict_score(self, context_vec: np.ndarray, action_vec_list: List[np.ndarray]) -> np.ndarray:
        """
        Predicts a score (expected reward) for each action given a context.
        """
        scores = []
        if not self.model_ready:
            print("[RL] Model not trained. Returning random scores.")
            return np.random.rand(len(action_vec_list))
            
        # If model is real (trained)
        for action_vec in action_vec_list:
            # Combine context (1, 12) and action (1, 8) -> (1, 20)
            combined_features = np.concatenate([context_vec, action_vec], axis=1)
            score = self.model.predict(combined_features)
            scores.append(score[0])
        return np.array(scores)

def load_bandit_model(model_path="bandit_model.pkl") -> MockLinUCB:
    """Loads the bandit model from disk, or creates a new one."""
    if os.path.exists(model_path):
        print("[RL] Loading existing bandit model from disk.")
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                if model.n_features != N_FEATURES:
                     print(f"[RL] Model feature mismatch! Expected {N_FEATURES}, got {model.n_features}. Creating new model.")
                     return MockLinUCB(n_features=N_FEATURES)
                model.model_ready = True
                return model
        except Exception as e:
            print(f"[RL] Error loading model: {e}. Creating new model.")
            
    print("[RL] No model found or loading failed. Creating new model.")
    return MockLinUCB(n_features=N_FEATURES)

# --- Feature Generation (Shared by Agent & Training) ---

# The 7 categories from your database/db.py 'apps' table
CATEGORY_LIST = ['songs', 'entertainment', 'socialmedia', 'games', 'communication', 'help', 'other']
CATEGORY_MAP = {name: i for i, name in enumerate(CATEGORY_LIST)}

def get_context_features() -> (Dict[str, Any], np.ndarray):
    """Generates a feature dictionary and vector for the current context."""
    now = datetime.now()
    
    time_of_day = "night"
    if 5 <= now.hour < 12: time_of_day = "morning"
    elif 12 <= now.hour < 18: time_of_day = "afternoon"
        
    day_of_week = "weekend" if now.weekday() >= 5 else "weekday"
    
    context_dict = {"time_of_day": time_of_day, "day_of_week": day_of_week}
    
    time_vec = [1.0 if time_of_day == "morning" else 0.0,
                1.0 if time_of_day == "afternoon" else 0.0,
                1.0 if time_of_day == "night" else 0.0]
    
    day_vec = [1.0 if day_of_week == "weekday" else 0.0,
               1.0 if day_of_week == "weekend" else 0.0]
    
    emotion_vec = [0.0] * 7 # Placeholder
    
    final_vec = np.array(emotion_vec + time_vec + day_vec).reshape(1, -1)
    return context_dict, final_vec

def get_action_features(app: AppRecommendation) -> (Dict[str, Any], np.ndarray):
    """Generates a feature dictionary and vector for a given action."""
    action_category = 'other' # default
    for cat_name in CATEGORY_MAP.keys():
        if cat_name.lower() in app.app_name.lower():
            action_category = cat_name
            break
    cat_lower = action_category
    
    action_dict = {
        "app_name": app.app_name,
        "is_local": app.is_local,
        "category": cat_lower
    }
    
    # [is_local, cat1, cat2, cat3, cat4, cat5, cat6, cat7]
    cat_vec = [0.0] * len(CATEGORY_LIST)
    cat_index = CATEGORY_MAP.get(cat_lower, CATEGORY_MAP['other']) # Default to 'other'
    cat_vec[cat_index] = 1.0
    
    features = [1.0 if app.is_local else 0.0] + cat_vec
    return action_dict, np.array(features).reshape(1, -1)

def featurize_context(context_dict: dict) -> np.ndarray:
    """Converts a context dict from DB into a feature vector."""
    emotion_list = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    emotion_map = {name: i for i, name in enumerate(emotion_list)}
    
    emotion_vec = [0.0] * 7
    emotion_index = emotion_map.get(context_dict.get('emotion', 'neutral'), 6)
    emotion_vec[emotion_index] = 1.0
    
    time_of_day = context_dict.get('time_of_day', 'night')
    time_vec = [1.0 if time_of_day == "morning" else 0.0,
                1.0 if time_of_day == "afternoon" else 0.0,
                1.0 if time_of_day == "night" else 0.0]
    
    day_of_week = context_dict.get('day_of_week', 'weekday')
    day_vec = [1.0 if day_of_week == "weekday" else 0.0,
               1.0 if day_of_week == "weekend" else 0.0]
               
    return np.array(emotion_vec + time_vec + day_vec)

def featurize_action(action_dict: dict) -> np.ndarray:
    """Converts an action dict from DB into a feature vector."""
    cat_lower = action_dict.get('category', 'other').lower()

    cat_vec = [0.0] * len(CATEGORY_LIST)
    cat_index = CATEGORY_MAP.get(cat_lower, CATEGORY_MAP['other'])
    cat_vec[cat_index] = 1.0
    
    features = [1.0 if action_dict.get('is_local', False) else 0.0] + cat_vec
    return np.array(features)