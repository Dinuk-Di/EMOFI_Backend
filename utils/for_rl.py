import ctypes
import winsound
import json
from datetime import datetime
import tkinter as tk
import winsound
from utils.rl_model import featurize_context, featurize_action, CATEGORY_MAP

# def show_feedback_dialog(title, message):
#     """
#     Shows a blocking Windows message box with YES/NO buttons.
#     Returns True if YES is clicked, False otherwise.
#     """
#     MB_YESNO = 0x04  # Yes/No buttons
#     IDYES = 6        # Return value for YES
    
#     winsound.MessageBeep() # Play a sound
#     result = ctypes.windll.user32.MessageBoxW(0, message, title, MB_YESNO)
    
#     return result == IDYES

def show_feedback_dialog(title, message):
    """
    Shows a custom blocking dialog in the bottom-right corner with YES/NO buttons.
    Returns True if YES is clicked, False otherwise.
    """
    # 1. Play Sound
    try:
        winsound.MessageBeep()
    except:
        pass

    # Container to store the result
    user_result = {"value": False}

    # --- Styling Constants ---
    bg_color = "#2b2b2b"
    text_color = "white"
    accent_yes = "#0e3aa9"  # Blue for YES
    accent_no = "#333333"   # Darker for NO

    # --- Event Handlers ---
    def on_yes():
        user_result["value"] = True
        root.destroy()

    def on_no():
        user_result["value"] = False
        root.destroy()
    
    # Hover effects
    def on_enter(e, color):
        e.widget['bg'] = color
    def on_leave(e, color):
        e.widget['bg'] = color

    # --- Window Setup ---
    root = tk.Tk()
    root.title(title)
    root.overrideredirect(True)      # Remove window frame/title bar
    root.attributes("-topmost", True) # Keep on top
    root.configure(bg=bg_color)

    # --- Positioning (Bottom Right) ---
    width, height = 300, 160
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # 20px from right, 70px from bottom (Taskbar clearance)
    x = screen_width - width - 20
    y = screen_height - height - 70
    
    root.geometry(f"{width}x{height}+{x}+{y}")

    # --- UI Elements ---
    
    # Title Bar (Small Close Button)
    top_bar = tk.Frame(root, bg=bg_color)
    top_bar.pack(fill="x", padx=10, pady=5)
    
    # Title Text
    tk.Label(root, text=title, bg=bg_color, fg="#4CAF50", 
             font=("Segoe UI", 12, "bold"), anchor="w").pack(fill="x", padx=20, pady=(0, 10))

    # Message Text
    tk.Label(root, text=message, bg=bg_color, fg=text_color, 
             font=("Segoe UI", 10), wraplength=260, justify="left", anchor="w").pack(fill="both", expand=True, padx=20)

    # Buttons Frame
    btn_frame = tk.Frame(root, bg=bg_color)
    btn_frame.pack(fill="x", padx=20, pady=20)

    # NO Button
    btn_no = tk.Button(btn_frame, text="No", width=10, 
                       bg=accent_no, fg="white", font=("Segoe UI", 9),
                       relief="flat", cursor="hand2", command=on_no)
    # Added ipady=5 to make the button taller
    btn_no.pack(side="right", padx=(10, 0), ipady=5)
    
    # YES Button
    btn_yes = tk.Button(btn_frame, text="Yes", width=10, 
                        bg=accent_yes, fg="white", font=("Segoe UI", 9, "bold"),
                        relief="flat", cursor="hand2", command=on_yes)
    # Added ipady=5 here as well
    btn_yes.pack(side="right", ipady=5)

    # Bind Hover Effects
    btn_yes.bind("<Enter>", lambda e: on_enter(e, "#164bc4")) # Lighter blue
    btn_yes.bind("<Leave>", lambda e: on_leave(e, accent_yes))
    
    btn_no.bind("<Enter>", lambda e: on_enter(e, "#4d4d4d"))  # Lighter gray
    btn_no.bind("<Leave>", lambda e: on_leave(e, accent_no))

    # --- Block Execution ---
    root.mainloop()
    
    return user_result["value"]



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