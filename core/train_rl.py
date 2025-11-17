import pickle
import json
import numpy as np
from sklearn.linear_model import Ridge
from contextlib import contextmanager

# --- Feature Dimensions (must match agent.py) ---
CONTEXT_DIM = 12
ACTION_DIM = 5
N_FEATURES = CONTEXT_DIM + ACTION_DIM

# --- Mock DB access ---
@contextmanager
def get_connection():
    yield "mock_connection"

def get_all_rl_experiences(conn):
    """Mock DB call to get all logged experiences."""
    # In a real system:
    # CURSOR.execute("SELECT context_json, action_json, reward FROM rl_experiences")
    # return CURSOR.fetchall()
    
    # Simulating data that would be in the DB
    return [
        ('{"emotion": "sad", "time_of_day": "afternoon", "day_of_week": "weekday"}',
         '{"app_name": "Spotify", "is_local": true, "category": "music"}',
         120.5),
        ('{"emotion": "angry", "time_of_day": "night", "day_of_week": "weekday"}',
         '{"app_name": "Poki", "is_local": false, "category": "game"}',
         250.0),
        ('{"emotion": "sad", "time_of_day": "morning", "day_of_week": "weekend"}',
         '{"app_name": "YouTube", "is_local": false, "category": "video"}',
         180.0)
    ]

# --- Feature Engineering (must match agent.py) ---

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
    cat = action_dict.get('category', 'other').lower()
    features = [
        1.0 if action_dict.get('is_local', False) else 0.0,
        1.0 if cat == 'music' else 0.0,
        1.0 if cat == 'video' else 0.0,
        1.0 if cat == 'game' else 0.0,
        1.0 if cat not in ['music', 'video', 'game'] else 0.0
    ]
    return np.array(features)

# --- Main Training Function ---

def train_model():
    print("Starting RL model training...")
    
    # 1. Load Data
    X = [] # Feature vectors (Context + Action)
    y = [] # Rewards (Dwell Time)
    
    with get_connection() as conn:
        experiences = get_all_rl_experiences(conn)
        
    if not experiences:
        print("No experiences found. Exiting.")
        return

    print(f"Loaded {len(experiences)} experiences from database.")
    
    # 2. Featurize Data
    for context_json, action_json, reward in experiences:
        try:
            context_dict = json.loads(context_json)
            action_dict = json.loads(action_json)
            
            context_vec = featurize_context(context_dict)
            action_vec = featurize_action(action_dict)
            
            # Combine into a single feature vector
            combined_features = np.concatenate([context_vec, action_vec])
            
            X.append(combined_features)
            y.append(reward)
        except Exception as e:
            print(f"Skipping malformed data: {e}")
            
    if not X:
        print("No valid data to train on. Exiting.")
        return
        
    X = np.array(X)
    y = np.array(y)
    
    # 3. Train Model
    # We use Ridge Regression, which is the core of LinUCB.
    # It learns to predict Reward = f(Context, Action)
    print(f"Training Ridge Regression model on {X.shape[0]} samples...")
    
    # `alpha` is the regularization strength (like LinUCB's alpha)
    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X, y)
    
    print("Training complete.")
    
    # 4. Save Model
    # We save the *trained regressor* inside our mock class
    # to be loaded by the agent.
    
    # Re-using the MockLinUCB class from the agent for persistence
    class MockLinUCB:
        def __init__(self, n_features):
            self.n_features = n_features
            self.model_ready = False
            self.model = None

        def predict_score(self, context_vec, action_vec_list):
            scores = []
            if not self.model_ready:
                return np.random.rand(len(action_vec_list))
                
            for action_vec in action_vec_list:
                combined_features = np.concatenate([context_vec, action_vec], axis=1)
                score = self.model.predict(combined_features)
                scores.append(score[0])
            return np.array(scores)

    
    final_bandit = MockLinUCB(n_features=N_FEATURES)
    final_bandit.model = model
    final_bandit.model_ready = True
    
    model_path = "bandit_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(final_bandit, f)
        
    print(f"Successfully saved updated model to {model_path}")