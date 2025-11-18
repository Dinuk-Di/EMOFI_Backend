import sys
import json
import pickle
import numpy as np
from sklearn.linear_model import Ridge

# Import your model class and constants
from .rl_model import MockLinUCB, load_bandit_model, N_FEATURES

def load_training_data(filepath="rl_training_data.jsonl"):
    """Loads the logged data and prepares it for scikit-learn."""
    X = []
    y = []
    
    try:
        with open(filepath, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # Recreate the (Context + Action) feature vector
                    context_vec = np.array(data['context'])
                    action_vec = np.array(data['action'])
                    
                    # Ensure they are flat before combining
                    combined_features = np.concatenate([
                        context_vec.flatten(), 
                        action_vec.flatten()
                    ])
                    
                    if combined_features.shape[0] == N_FEATURES:
                        X.append(combined_features)
                        y.append(data['reward'])
                    else:
                        print(f"Skipping data point: feature mismatch. Expected {N_FEATURES}, got {combined_features.shape[0]}")
                        
                except Exception as e:
                    print(f"Skipping bad line: {e}")
                    
    except FileNotFoundError:
        print("No training data file found.")
        return None, None
        
    if not X:
        return None, None
        
    return np.array(X), np.array(y)

def train_and_save_model():
    """Main training function."""

    # --- 1. READ ARGUMENTS FROM SUBPROCESS CALL ---
    if len(sys.argv) < 3:
        print("[Trainer] FATAL ERROR: Missing log_file_path or model_file_path.")
        return

    log_file_path = sys.argv[1]
    model_file_path = sys.argv[2]
    # --- END OF CHANGE ---

    print("[Trainer] Loading training data...")
    print(f"[Trainer] Loading data from: {log_file_path}") # For logging

    # --- 2. PASS PATH TO LOADER ---
    X_train, y_train = load_training_data(log_file_path)

    if X_train is None or len(y_train) < 10:
        print(f"[Trainer] Not enough data to train. Need at least 10 samples, found {len(y_train) if y_train is not None else 0}.")
        return

    print(f"[Trainer] Training on {len(y_train)} data points.")

    # --- 3. ADD THIS BLOCK BACK ---
    # This is the missing logic that defines 'rl_model_container'
    
    # 3a. Create the "container" class
    print("[Trainer] Creating new model container.")
    rl_model_container = MockLinUCB(n_features=N_FEATURES)
    
    # 3b. Create and train the new Ridge regression model
    print("[Trainer] Fitting Ridge model...")
    new_ridge_model = Ridge()
    new_ridge_model.fit(X_train, y_train)
    print("[Trainer] Model fitting complete.")
    
    # 3c. Put the trained model into our container
    rl_model_container.model = new_ridge_model
    rl_model_container.model_ready = True
    # --- END OF MISSING BLOCK ---


    # --- 4. USE MODEL PATH WHEN SAVING ---
    try:
        # This line will now work because 'rl_model_container' is defined
        with open(model_file_path, "wb") as f:
            pickle.dump(rl_model_container, f)

        print(f"[Trainer] New model trained and saved successfully to '{model_file_path}'!")

    except Exception as e:
        print(f"[Trainer] Error saving model: {e}")

if __name__ == "__main__":
    train_and_save_model()