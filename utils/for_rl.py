import ctypes
import winsound
import json
from datetime import datetime

# --- (Assume your RL model code is in a file named 'rl_model_utils.py') ---
# You must import the featurizing functions from your *other* file.
from utils.rl_model import featurize_context, featurize_action, CATEGORY_MAP

def show_feedback_dialog(title, message):
    """
    Shows a blocking Windows message box with YES/NO buttons.
    Returns True if YES is clicked, False otherwise.
    """
    MB_YESNO = 0x04  # Yes/No buttons
    IDYES = 6        # Return value for YES
    
    winsound.MessageBeep() # Play a sound
    result = ctypes.windll.user32.MessageBoxW(0, message, title, MB_YESNO)
    
    return result == IDYES

def log_training_data(context_dict, action_object, reward, filepath):
    """
    Saves the context, action, and reward to a file for later training.
    
    Args:
        context_dict (dict): The context features (e.g., {'emotion': 'Sad', ...})
        action_object (AppRecommendation): The Pydantic model of the chosen action.
        reward (int): 1 for good, 0 for bad.
    """
    try:
        # 1. Convert context dict to feature vector
        # This is now correct because context_dict is the right argument
        context_vec = featurize_context(context_dict)

        # --- 2. Handle Action featurizing ---
        # We now correctly receive an AppRecommendation object
        
        # A simple (but imperfect) way to get category from app_name
        action_category = 'other' # default
        for cat_name in CATEGORY_MAP.keys():
            # -----------------------------------------------------------------
            # FIXED: Use dot notation (action_object.app_name) instead of .get()
            # -----------------------------------------------------------------
            if cat_name.lower() in action_object.app_name.lower():
                action_category = cat_name
                break
                
        # Build the simple dict that featurize_action expects
        simple_action_dict = {
            # We use our guessed category
            "category": action_category,
            # -----------------------------------------------------------------
            # FIXED: Use dot notation (action_object.is_local) instead of .get()
            # -----------------------------------------------------------------
            "is_local": action_object.is_local 
        }

        # Featurize the simplified action dict
        action_vec = featurize_action(simple_action_dict)
        
        # 3. Create the data point
        data_point = {
            # Save vectors as plain lists
            "context": context_vec.tolist(), 
            "action": action_vec.tolist(),
            "reward": reward
        }
        
        # 4. Append to a JSON Lines file (each line is a valid JSON)
        with open(filepath, "a") as f:
            f.write(json.dumps(data_point) + "\n")
            
        print(f"[RL] Logged data point: reward={reward}")
        
    except Exception as e:
        print(f"[RL] Error logging training data: {e}")