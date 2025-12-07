from typing import Dict, List, Optional, Any
import base64
from collections import Counter
import re
import time
from langgraph.graph import StateGraph, END
import requests
from utils.desktop import capture_desktop
from ui.notification import send_notification
import ctypes
from collections import Counter
from typing import List, Optional
from pydantic import BaseModel
from utils.tools import open_recommendations
from database.db import get_apps, get_connection,add_agent_recommendations
from dotenv import load_dotenv
import os
import json
from pydantic import BaseModel, Field
import ctypes
from openai import OpenAI
from old_utils.state import app_state, pickle_save
import winsound
from datetime import datetime
import subprocess 
import sys    
import socket 
import asyncio
from desktop_notifier import DesktopNotifier, Button
from pathlib import Path
import os
from threading import Event
from win11toast import toast
from utils.tools import resource_path

# If threads are used
import threading

# If ollama is used
import ollama

from utils.rl_model import load_bandit_model,featurize_context,get_action_features
from utils.for_rl import log_training_data, show_feedback_dialog
from utils.models import AppRecommendation

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# --- To connect with raspberry pi ---
# def send_signal_to_pi(signal_message: str):
#     """Sends a simple TCP signal to the Raspberry Pi."""
#     # Raspberry Pi's actual IP address
#     RPI_IP = '10.50.228.15'  
#     # THE SAME PORT AS THE PI SERVER
#     PORT = 65432         
    
#     try:
#         # Create a socket and connect
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:           
#             s.settimeout(2.0)
#             s.connect((RPI_IP, PORT))
            
#             s.sendall(signal_message.encode('utf-8'))
#             print(f"[Signal Sender] Sent signal to RPi: '{signal_message}'")
            
#             # Optional: Wait for a short acknowledgement from the Pi
#             # s.settimeout(1) # Set a timeout for receiving
#             # ack = s.recv(1024)
#             # print(f"[Signal Sender] RPi acknowledged: {ack.decode()}")

#     except ConnectionRefusedError:
#         print(f"[Signal Sender] ERROR: Connection refused. Is the RPi server running on {RPI_IP}:{PORT}?")
#     except Exception as e:
#         print(f"[Signal Sender] An error occurred while sending signal: {e}")


# --- New HTTP POST version ---
def send_signal_to_pi(payload: dict):
    """Sends an HTTP POST request to the Raspberry Pi."""
    # Raspberry Pi's actual IP address
    RPI_IP = '10.50.228.36'  
    # Update this to the port your HTTP server is listening on (e.g., 5000 for Flask)
    PORT = 5000         
    
    url = f"http://{RPI_IP}:{PORT}/api/notification"
    
    try:
        # Send the dictionary directly using the 'json' parameter
        # requests will automatically add Content-Type: application/json
        response = requests.post(url, json=payload, timeout=2.0)
        
        if response.status_code == 200:
            print(f"[Signal Sender] Successfully sent signal to RPi: {response.status_code}")
        else:
            print(f"[Signal Sender] RPi returned error: {response.status_code} - {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"[Signal Sender] ERROR: Could not connect to {url}. Is the RPi server running?")
    except Exception as e:
        print(f"[Signal Sender] An error occurred while sending signal: {e}")

# --- Agent System Implementation ---
def run_agent_system(emotions):
    initial_state = AgentState(
        emotions=emotions,
        average_emotion=None,
        continue_workflow=None,
        recommendation=None,
        recommendation_options= [],
        executed=False,
        action_executed=None,
        action_time_start=0,
        chosen_action=None,
        context_at_recommendation=None

    )
    agent_workflow = create_workflow()

    config = {"recursion_limit": 100}
    
    return agent_workflow.invoke(initial_state, config=config)


# class AppRecommendation(BaseModel):
#     app_name: str = Field(description="Name of recommended application")
#     app_url: str = Field(description="URL or local path of the application")
#     search_query: str = Field(description="Search query if web-based application")
#     is_local: bool = Field(default=False, description="Whether the app is a local executable")

class RecommendationResponse(BaseModel):
    recommendation: str = Field(description="4-word mood improvement suggestions")
    recommendation_options: List[AppRecommendation] = Field(description="Two app recommendations")
class RecommendationList(BaseModel):
    listofRecommendations: List[RecommendationResponse] = Field(description="List of 3 recommendations with options")

class AgentState(BaseModel):
    emotions: List[str]
    average_emotion: Optional[str]
    continue_workflow: Optional[bool]
    recommendation: Optional[List[str]]
    recommendation_options: Optional[List[List[AppRecommendation]]]
    executed: Optional[bool]
    action_executed: Optional[str]
    action_time_start: Optional[float]
    open_app_handle: Optional[Any] = None
    app_type: Optional[str] = None
    continue_waiting: Optional[bool] = None
    wait_start_time: Optional[float] = None

    chosen_action: Optional[AppRecommendation] = None           # The app the user clicked on
    context_at_recommendation: Optional[Dict] = None # The context (emotion, time)


def create_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("calculate_emotion", average_emotion_agent)
    workflow.add_node("interrupt_check", interrupt_check_agent)
    workflow.add_node("generate_recommendation", recommendation_agent)
    workflow.add_node("execute_action", task_execution_agent)
    workflow.add_node("wait_for_close", wait_for_close_agent)
    workflow.add_node("exit_action", task_exit_agent)
    
    workflow.set_entry_point("calculate_emotion")
    workflow.add_edge("calculate_emotion", "interrupt_check")
    workflow.add_conditional_edges(
        "interrupt_check",
        lambda state: "generate_recommendation" if state.continue_workflow else END,
    )
    workflow.add_edge("generate_recommendation", "execute_action")
    workflow.add_conditional_edges(
        "execute_action", 
        lambda state: "wait_for_close" if state.executed else END,
    )
    workflow.add_conditional_edges(
        "wait_for_close",
        lambda state: "wait_for_close" if state.continue_waiting else "exit_action",
    )
    workflow.add_edge("exit_action", END)
    return workflow.compile()



def average_emotion_agent(state):
    """Calculate most frequent emotion from AgentState model"""
    if not state.emotions:
        return {"average_emotion": "neutral"}
    print(f"[Agent] Emotions: {state.emotions}")
    counter = Counter(state.emotions)
    most_common = counter.most_common(1)[0][0]
    print(f"[Agent] Average emotion: {most_common}")

    app_state.averageEmotion = most_common
    pickle_save()

    return {"average_emotion": most_common}

# Remove reasoning tags from the response
def clean_think_tags(text):
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

# def show_notification_with_ok(title, message, duration=15):
#     """
#     Show Windows notification with OK button and wait for user response.
#     Returns True if OK clicked within duration, else False.
#     """
#     # Create the notification
#     toast = Notification(app_id="EMOFI", title=title, msg=message, duration="long")

#     # Add OK button that triggers a callback (using protocol)
#     toast.add_actions(label="OK")

#     # Show notification
#     toast.show()

#     # Wait for a certain time for the user to click (simulate by polling a flag)
#     clicked = {"status": False}

#     def monitor_click():
#         # Simulate action URL check
#         # Real-world: This needs a listener or log check
#         for i in range(duration):
#             time.sleep(1)
#             # Here you'd check if the user clicked (through action callback or system log)
#             # We'll simulate by checking a file or variable
#             if clicked["status"]:
#                 break

#     # Start monitoring in a separate thread
#     t = threading.Thread(target=monitor_click)
#     t.start()
#     t.join(timeout=duration)
#     print("[Agent] User clicked OK:", clicked["status"])

#     return clicked["status"]



# def interrupt_check_agent(state):
#     print("[Agent] Running interrupt_check_agent...")

#     # You could base this on the emotion if you want, or always send
#     emotion = state.average_emotion
    
#     negative_emotions = ["Angry", "Sad", "Fear", "Disgust", "Stress", "Boring"]
    
#     user_response = None

#     if emotion in negative_emotions:
#         # Show notification with OK button
#         user_response = show_notification_with_ok(
#             title="Your Emotion Is Not Good",
#             message="Shall we give some suggestions to boost your mood?",
#             duration=15  # Notification auto-dismiss after 15 sec
#         )

#         if not user_response:  # If user didn't click OK in time
#             print("[Agent] No user response, ending workflow.")
#             # End the workflow early by setting executed=True and returning END
#             return {"average_emotion": emotion, "executed": False} 
#     print(f"[Agent] Emotion is {emotion}")


#     if user_response is None or user_response == "No":
#         print("[Agent] User declined. Ending workflow.")
#         return {"continue_workflow": False}

#     print("[Agent] User accepted. Continuing workflow.")
#     return {"continue_workflow": True}
    

# def show_notification_with_ok(title, message):
#     """Show Windows message box with OK/Cancel buttons"""
#     MB_OKCANCEL = 0x01
#     IDOK = 1
#     winsound.MessageBeep()
#     result = ctypes.windll.user32.MessageBoxW(0, message, title, MB_OKCANCEL)
#     return result == IDOK


# Set App ID so Windows shows the correct name/icon
try:
    myappid = 'mycompany.emofi.app.1.0' 
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except Exception:
    pass


def show_notification_with_ok(title, message,timeout_seconds=120):
    icon_path = Path(resource_path("assets/res/Icon.ico")) 

    result_container = {"clicked": False}

    async def async_notification_logic():
        
        notifier = DesktopNotifier(app_name="EMOFI")
        
        user_interaction_event = asyncio.Event()

        def on_clicked_callback():
            result_container["clicked"] = True
            user_interaction_event.set()

        try:
            await notifier.request_authorisation()
        except:
            pass 

        # 3. SEND NOTIFICATION
        await notifier.send(
            title=title,
            message=message,
            on_clicked=on_clicked_callback,
            buttons=[Button("Yes, Proceed", on_clicked_callback)],
            icon=icon_path
        )

        try:
            await asyncio.wait_for(user_interaction_event.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            result_container["clicked"] = False

    try:
        asyncio.run(async_notification_logic())
    except Exception as e:
        print(f"[Notification] Error: {e}")
        return False

    return result_container["clicked"]


def interrupt_check_agent(state):
    print("[Agent] Running interrupt_check_agent...")
    emotion = state.average_emotion
    negative_emotions = ["Angry", "Sad", "Fear", "Disgust", "Stress", "Boring"]
    
    # Always reset continue_workflow to False
    state.continue_workflow = False
   
    
    if emotion in negative_emotions:
        print(f"[Agent] Negative emotion detected: {emotion}")

        try:
            emotion_detected = state.average_emotion.lower()            
            # Construct the custom message for the Pi to speak
            speech_message = (
                f"Hi there, you seem {emotion_detected}. "
                f"We are going to give you some recommendations to boost your mood. "
                "Please click OK on the notification to proceed."
            )
            
            # Create the structured JSON payload
            signal_payload = {
                "trigger": "recommendations_ready",
                "emotion": emotion_detected,
                "message": speech_message
            }
            
            # Send the JSON payload
            # send_signal_to_pi(json.dumps(signal_payload))  
            send_signal_to_pi(signal_payload)
        except Exception as e:
                print(f"[Agent] Failed to send signal to RPi: {e}")
    
        # Show blocking message box
        user_responded = show_notification_with_ok(
            "Your Emotion Is Not Good",
            "Shall we give some suggestions to boost your mood?"
        )
        
        if user_responded:
            print("[Agent] User accepted recommendations")
            state.continue_workflow = True
    else:
        print(f"[Agent] Neutral/positive emotion: {emotion}")

    return {"continue_workflow": state.continue_workflow}


def parse_llm_response(text):
    try:
        # Clean <think> tags if present
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # Extract recommendation
        rec_match = re.search(r'recommendation:\s*(.+)', text)
        recommendation = rec_match.group(1).strip() if rec_match else None

        # Extract recommendation_options (raw list inside [])
        options_match = re.search(r'recommendation_options:\s*\[(.*)\]', text, re.DOTALL)
        options_raw = options_match.group(1).strip() if options_match else ""

        # Convert (app_name: 'X', ...) → {"app_name": "X", ...}
        options = []
        for option_text in re.findall(r'\((.*?)\)', options_raw, re.DOTALL):
            entry = {}
            for kv in option_text.split(','):
                key, val = kv.split(':', 1)
                entry[key.strip()] = val.strip().strip("'\"")
            options.append(entry)

        return recommendation, options

    except Exception as e:
        print("[Agent] Error parsing LLM response block:", e)
        return None, []

def extract_json_from_text(text):
    try:
        # Find JSON between ```json and ```
        match = re.search(r'```json\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
        if match:
            return match.group(1)

        # If no code block, try to parse whole text
        if text.strip().startswith("{") or text.strip().startswith("["):
            return text.strip()

        raise ValueError("No valid JSON found in text.")
    except Exception as e:
        print("[Agent] JSON extraction failed:", e)
        return None
    
def recommendation_agent(state):
    emotion = state.average_emotion
    # task = state.detected_task
    print(f"[Agent] Processing for emotion={emotion!r}")

    negative_emotions = ["Angry", "Sad", "Fear", "Disgust", "Stress", "Boring"]
    if emotion not in negative_emotions:
        print("[Agent] Mood is fine – no recommendation.")
        return {"recommendation": ["No action needed"], "recommendation_options": []}

    conn = get_connection()
    if not conn:
        print("[Agent] DB connection failed – skip recommendation.")
        return {"recommendation": ["No action needed"], "recommendation_options": []}

    available_apps = get_apps(conn)
    print("[Agent] Available apps:", available_apps)
    prompt = f"""
            You are a recommendation engine.

            Context:
            - User feels: "{emotion}"
            - Available installed apps (format: category | name | app_id | path):
            {available_apps!r}

            Goal:
            Generate EXACTLY 5 mood-improvement suggestions, each consisting of:
            - recommendation: A phrase of exactly FOUR words.
            - recommendation_options: An array of EXACTLY 2 options per recommendation. Each option must include:
                - app_name: (string)
                - app_url: (either a valid HTTPS URL for web apps OR local file path or app_id for installed apps)
                - search_query: (string, required only for web apps)
                - is_local: (true if app is installed locally, false if web)
                - category: (string) One of: 'songs', 'entertainment', 'socialmedia', 'games', 'communication', 'help', 'other'

            STRICT RULES:
            1. Output ONLY valid JSON — no extra text, no explanations, no markdown.
            2. JSON format: An array of 3 objects with keys: recommendation, recommendation_options.
            3. Each recommendation must have TWO different apps (no duplicates across or within).
            4. Prefer local apps over web apps if available.
            5. For web apps:
            - All URLs must start with "https://".
            - Use "<search_query>" placeholder in the app_url instead of inserting actual query.
            - Example web apps are YouTube, Spotify, Online Game (https://poki.com/), MyFlixer (https://myflixerz.to/).
            6. For local apps:
            - Use given path as app_url and set is_local = true.
            - search_query is empty
            7. Don't use same app in multiple recommendations.
            8. Each recommendation must be exactly 4 words, meaningful, and mood-impro
            9. You MUST provide a 'category' for every app. For local apps, use their category from the list. For web apps, choose the best fit (e.g., 'songs' for Spotify, 'entertainment' for YouTube).
            

            Example of expected structure (do NOT include this in response):
            [
            {{
                "recommendation": "Take a quick break",
                "recommendation_options": [
                {{
                    "app_name": "Spotify",
                    "app_url": "https://open.spotify.com/search/<search_query>",
                    "search_query": "relaxing music",
                    "is_local": false
                }},
                {{
                    "app_name": "KMPlayer",
                    "app_url": "C:\\\\Program Files\\\\KMPlayer 64X\\\\KMPlayer.exe",
                    "search_query": "",
                    "is_local": true
                }}
                ]
            }}
            ]

            Now, produce the final JSON output:
            """


    full_schema = RecommendationList.model_json_schema()

    try:
        # res = requests.post(
        #     url="https://openrouter.ai/api/v1/chat/completions",
        #     headers={
        #         "Authorization": f"Bearer {QWEN_API_KEY}",
        #         "Content-Type": "application/json"
        #     },
        #     data=json.dumps({
        #         "model": "deepseek/deepseek-r1-0528-qwen3-8b:free",
        #         "messages": [
        #             {"role": "system", "content": "You are an assistant. Output must be valid JSON only."},
        #             {"role": "user", "content": prompt}
        #         ],
        #         "response_format": {
        #             "type": "json_schema",
        #             "json_schema": {
        #                 "name": "recommendation_list",
        #                 "strict": True,
        #                 "schema": full_schema
        #             }
        #         },
        #         "structured_outputs": True
        #     }
        # ))

        schema = {
            "type": "object",
                    "properties": {
                        "listofRecommendations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "recommendation": {"type": "string"},
                                    "recommendation_options": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "app_name": {"type": "string"},
                                                "app_url": {"type": "string"},
                                                "search_query": {"type": "string"},
                                                "is_local": {"type": "boolean"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
            }

        client = OpenAI()
        
        response = client.responses.parse(
            model="gpt-4o-2024-08-06",
            input=[
                {"role": "system", "content": "Give the proper structured output."},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            text_format=RecommendationList,
        )

        print("[Agent] API response:", response.output_parsed)

        resp_data = response.output_parsed

        # Extract recommendations
        recommendations_list = [rec.recommendation for rec in resp_data.listofRecommendations]
        recommendation_options_list = [rec.recommendation_options for rec in resp_data.listofRecommendations]

        print("Final Recommendations:", recommendations_list)
        print("Options:", recommendation_options_list)     

        
        # Update state
        state.recommendation = recommendations_list
        state.recommendation_options = recommendation_options_list

        try:
            for i, recommendation_type in enumerate(recommendations_list):
                for option in recommendation_options_list[i]:
                    recommed_app = option.app_name
                    app_url = option.app_url
                    search_query = option.search_query
                    is_local = option.is_local

                    add_agent_recommendations(
                        conn,
                        1,
                        recommendation_type,
                        recommed_app,
                        app_url,
                        search_query,
                        is_local
                    )
        except Exception as e:
            print("[Agent] Error adding recommendations to DB:", e)


        return {
            "recommendation": recommendations_list,
            "recommendation_options": recommendation_options_list
        }

    except Exception as e:
        print("[Agent] Unexpected error:", e)
        return {
            "recommendation": ["No action needed"],
            "recommendation_options": [],
            "listofRecommendations": RecommendationList(listofRecommendations=[])
        }



def send_blocking_message(title, message):
    MB_OK = 0x0
    ctypes.windll.user32.MessageBoxW(0, message, title, MB_OK)



def task_execution_agent(state):
    recommended_output = state.recommendation
    recommended_options = state.recommendation_options
    average_emotion = state.average_emotion
    
    # Ensure we have recommendations to process
    if not recommended_output or "No action needed" in recommended_output:
        return {"executed": False}
    if "No action needed" not in recommended_output:
        state.executed = True     

    # --- RL logic starts here ---

    # --- 1. CAPTURE CURRENT CONTEXT ---
    # (This logic is from your rl_model_utils.py file)
        now = datetime.now()
        time_of_day = "night"
        if 5 <= now.hour < 12: time_of_day = "morning"
        elif 12 <= now.hour < 18: time_of_day = "afternoon"
        day_of_week = "weekend" if now.weekday() >= 5 else "weekday"

        
        # This is the full context dictionary
        current_context_dict = {
            "emotion": average_emotion, 
            "time_of_day": time_of_day, 
            "day_of_week": day_of_week
        }

        state.context_at_recommendation = current_context_dict

        print("[RL] Current context for ranking:", current_context_dict)


        # --- 2. LOAD RL MODEL AND SCORE RECOMMENDATIONS ---
        # This is the "ranking" step you mentioned
        
        # We must flatten the list of lists: [[App1, App2], [App3, App4], ...]
        all_apps_to_rank = [app for sublist in recommended_options for app in sublist]
        
        try:
            model_file_path = resource_path("bandit_model.pkl")

            # 3. Load the model using the full, absolute path
            print(f"[RL] Loading model from: {model_file_path}")
            rl_model = load_bandit_model(model_file_path)
            # --- END: REPLACEMENT BLOCK ---
            
            if rl_model.model_ready:
                print("[RL] Ranking apps with trained model...")
                context_vec = featurize_context(current_context_dict).reshape(1, -1)
                
                # Get action vectors for all apps
                action_vec_list = [get_action_features(app)[1].reshape(1, -1) for app in all_apps_to_rank]
                
                # Get a score for each app
                scores = rl_model.predict_score(context_vec, action_vec_list)
                
                # Pair apps with their scores and sort
                scored_apps = sorted(zip(all_apps_to_rank, scores), key=lambda x: x[1], reverse=True)
                
                # --- This is your Top 3 logic ---
                top_3_apps = [app for app, score in scored_apps[:3]]
                
                # We also need the original "recommendation" text
                # This is complex. For simplicity, let's just create 3 new "recommendations"
                recommended_output_ranked = [f"Suggestion {i+1}" for i in range(len(top_3_apps))]
                # Group them back into lists of 2 (or just 1)
                recommended_options_ranked = [[app] for app in top_3_apps]

                print(f"[RL] Ranked Apps (Top 3): {[app.app_name for app in top_3_apps]}")
                
            else:
                print("[RL] Model not trained, using first 3 apps from LLM.")
                # Fallback if model isn't trained: just take the first 3 apps
                recommended_output_ranked = recommended_output[:3]
                recommended_options_ranked = recommended_options[:3]

        except Exception as e:
            print(f"[RL] Error during ranking: {e}. Using LLM default.")
            recommended_output_ranked = recommended_output[:3]
            recommended_options_ranked = recommended_options[:3]

            # RL logic ends here
    
        
    chosen_recommendation = send_notification(
        "Recommendations by EMOFI", 
        recommended_output,
        recommended_options
    )

    print("Chosen recommendation at task_execute_agent: ", chosen_recommendation)   
    
    
    # Handle case where user didn't select anything
    if not chosen_recommendation:
        state.chosen_action = False
        state.executed = False
        return {"executed": False}
        

    state.chosen_action = chosen_recommendation

    print("After setting chosen action in state:", state.chosen_action)
        
    # Get open results safely
    result = open_recommendations(chosen_recommendation)
    if not result:
        state.executed = False
        return {"executed": False}
        
    is_opened, app_handle, app_type = result
    print("Result from open_recommendations:", is_opened, app_handle, app_type)
    
    # Update state only if app was opened
    if is_opened:
        return {
            "executed": True,
            "open_app_handle": app_handle,
            "app_type": app_type,
            "continue_waiting": True,
            "wait_start_time": time.time(),
            "chosen_action": chosen_recommendation,
            "context_at_recommendation": current_context_dict
        }
    
    return {"executed": False}


import psutil

def wait_for_close_agent(state):
    MAX_WAIT_SECONDS = 300  # 5 minute timeout
    
    # Check if we should stop waiting
    if not state.continue_waiting or not state.open_app_handle:
        return {"continue_waiting": False}
    
    # Check timeout
    elapsed = time.time() - state.wait_start_time
    if elapsed > MAX_WAIT_SECONDS:
        print(f"[Agent] Wait timeout after {MAX_WAIT_SECONDS} seconds")
        return {
            "continue_waiting": False,
            "open_app_handle": None,
            "app_type": None
        }
        
    # Check if app is closed
    app_closed = False
    
    if state.app_type == 'local':
        try:
            process = psutil.Process(state.open_app_handle)
            app_closed = not process.is_running()
        except psutil.NoSuchProcess:
            app_closed = True
            
    elif state.app_type == 'web':
        try:
            # This will throw if browser closed
            state.open_app_handle.current_url
        except Exception:
            app_closed = True
    
    # Update waiting status
    if app_closed:
        print("[Agent] Detected app closure")
        return {
            "continue_waiting": False,
            "open_app_handle": None,
            "app_type": None
        }
    
    # Wait before checking again
    time.sleep(5)  # Increased sleep to reduce recursion
    print(f"[Agent] Still waiting for app to close ({int(elapsed)}s elapsed)")
    return {"continue_waiting": True}

# def task_exit_agent(state):
#     task_executed = True
#     if not state.executed:
#         return {"executed": False, "action_time_start": None}
#     print("Thread is running")
#     while task_executed:
#         time.sleep(50)
#         task_executed = False
#     print("Thread is closed")
#     return {"executed": False, "action_time_start": None}

def task_exit_agent(state):
    chosen_action = state.chosen_action
    context_at_recommendation = state.context_at_recommendation
    executed = state.executed

    print("[Agent] Executing task_exit_agent...")
    print(f"[Agent] State before exit: executed={executed}, chosen_action={chosen_action}, context={context_at_recommendation}")

    # --- THIS IS THE NEW FEEDBACK LOGIC ---
    if executed and chosen_action:
        print("[Agent] App closed. Asking for feedback.")
        
        # 1. Ask for feedback
        user_said_yes = show_feedback_dialog(
            "Feedback",
            "Did this help to improve your mood?"
        )
        
        # 2. Assign reward
        reward = 1 if user_said_yes else 0
        print(f"[Agent] User feedback: {'Yes' if user_said_yes else 'No'} (Reward: {reward})")
          # <-- 2nd: The AppRecommendation OBJECT       
        
        try:
            print("[Agent] Preparing to log data and trigger training...")

            log_file_path = resource_path("rl_training_data.jsonl")
            model_file_path = resource_path("bandit_model.pkl")
            # model_file_path = os.path.join(backend_folder_path, "bandit_model.pkl")

            # --- 4. Log Data (using the new path) ---
            log_training_data(
                context_at_recommendation,  # The context DICT
                chosen_action,              # The AppRecommendation OBJECT
                reward,
                log_file_path               # Pass the full path to the logger
            )

            # --- 5. Trigger Background Training (using new paths) ---
            python_exe = sys.executable 
            cmd_with_args = [
                python_exe,
                '-m', 'Backend.utils.train_rl_model',
                log_file_path,
                model_file_path
            ]
            
            print(f"[Agent] Starting subprocess...")

            # We still run from the 'project_root' (DesktopApp),
            # because that's where 'python -m Backend...' needs to run from.
            with open("trainer_stdout.log", "wb") as out, open("trainer_stderr.log", "wb") as err:
                subprocess.Popen(cmd_with_args, stdout=out, stderr=err, cwd=os.path.dirname(os.path.dirname(log_file_path)))
            
            print("[Agent] Training process started in background. Check 'trainer_stdout.log' for details.")

        except Exception as e:
            # Catch any errors during the logging/training process
            print(f"[Agent] Error during logging/training step: {e}")
        # ----------------------------------------------------
        # ✅ END: REPLACE YOUR AUTO-TRAIN BLOCK WITH THIS
    
    # --- Reset State (This part is unchanged) ---
    print("[Agent] Resetting state and finishing workflow.")
    return {
        "executed": False,
        "action_time_start": None,
        "open_app_handle": None,
        "app_type": None,
        "continue_waiting": None,
        "wait_start_time": None,
        "chosen_action": None,
        "context_at_recommendation": None
    }


# def send_blocking_message(title, message):
#     MB_OK = 0x0
#     ctypes.windll.user32.MessageBoxW(0, message, title, MB_OK)

# def task_execution_agent(state):
#     recommendation = state.recommendation
#     if "No action" in recommendation:
#         return {"executed": False}

#     send_blocking_message(
#         title="Emotion Assistant",
#         message=f"You seem {state.average_emotion}. Recommendation: {recommendation}"
#     )
#     # This line runs only after user presses OK in the message box
#     execute_task(recommendation)
#     return {"executed": True}


def task_detection_agent(state):
    try:
        if state.average_emotion == "Neutral" or state.average_emotion == "Happy" or state.average_emotion == "Surprise":
            print("[Agent] No task detection needed for neutral emotion.")
            return {"detected_task": "No Need to Detect Task"}
        # Capture screenshot as a base64 string (possibly with prefix)
        screenshot = capture_desktop()
        if not screenshot:
            raise ValueError("Failed to capture screenshot")
        # Remove data URI prefix if present
        if screenshot.startswith('data:image'):
            screenshot = screenshot.split(',')[1]

        # Validate base64 string (optional, for debugging)
        try:
            base64.b64decode(screenshot)
        except Exception as decode_err:
            raise ValueError(f"Invalid base64 screenshot: {decode_err}")

        # Send the raw base64 string (no prefix) to Ollama
        # response = ollama.generate(
        #     model="llava:7b",
        #     prompt="Describe user's current activity. Focus on software and tasks.",
        #     images=[screenshot]
        # )
        headers = {
            "Connection": "close",  # Disable keep-alive
            "Content-Type": "application/json"
        }
        response = requests.post(
            "https://d53cb0fd37cb.ngrok-free.app/api/generate",
            headers=headers,
            json={
                "model": "llava:7b",
                "prompt": "Describe user's current activity. Focus on software and tasks.",
                "images": [screenshot],
                "stream": False
            }
        )

        # Handle HTTP errors
        if response.status_code != 200:
            print(f"API error ({response.status_code}): {response.text[:100]}...")
            return {"detected_task": "unknown"}

        # Parse JSON response
        response_data = response.json()
        detected_task = response_data.get('response', '').strip()
        state.detected_task = detected_task
        print(f"Detected task: {detected_task}")
        return {"detected_task": detected_task}

    except Exception as e:
        print(f"Error detecting task: {str(e)}")
        return {"detected_task": "unknown"}