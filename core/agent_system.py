from typing import Dict, List, Optional, Any
import base64
from collections import Counter
import re
import time
from langgraph.graph import StateGraph, END
import requests
from utils.desktop import capture_desktop
from ui.notification import send_notification, execute_task
import ollama
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
from winotify import Notification
import threading
import ctypes
from openai import OpenAI
import win32gui
from old_utils.state import app_state, pickle_save, pickle_load
import winsound

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def run_agent_system(emotions):
    initial_state = AgentState(
        emotions=emotions,
        average_emotion=None,
        continue_workflow=None,
        recommendation=None,
        recommendation_options= [],
        executed=False,
        action_executed=None,
        action_time_start=0
    )
    agent_workflow = create_workflow()
    
    # Increase recursion limit
    config = {"recursion_limit": 100}  # Allow up to 100 steps
    
    return agent_workflow.invoke(initial_state, config=config)

class AppRecommendation(BaseModel):
    app_name: str = Field(description="Name of recommended application")
    app_url: str = Field(description="URL or local path of the application")
    search_query: str = Field(description="Search query if web-based application")
    is_local: bool = Field(default=False, description="Whether the app is a local executable")

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
    wait_start_time: Optional[float] = None  # Track when waiting began


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
    

def show_notification_with_ok(title, message):
    """Show Windows message box with OK/Cancel buttons"""
    MB_OKCANCEL = 0x01
    IDOK = 1
    winsound.MessageBeep()
    result = ctypes.windll.user32.MessageBoxW(0, message, title, MB_OKCANCEL)
    return result == IDOK

def interrupt_check_agent(state):
    print("[Agent] Running interrupt_check_agent...")
    emotion = state.average_emotion
    negative_emotions = ["Angry", "Sad", "Fear", "Disgust", "Stress", "Boring"]
    
    # Always reset continue_workflow to False
    state.continue_workflow = False
    
    if emotion in negative_emotions:
        print(f"[Agent] Negative emotion detected: {emotion}")
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
            Generate EXACTLY 3 mood-improvement suggestions, each consisting of:
            - recommendation: A phrase of exactly FOUR words.
            - recommendation_options: An array of EXACTLY 2 options per recommendation. Each option must include:
                - app_name: (string)
                - app_url: (either a valid HTTPS URL for web apps OR local file path or app_id for installed apps)
                - search_query: (string, required only for web apps)
                - is_local: (true if app is installed locally, false if web)

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

# def task_execution_agent(state):
#     recommended_output = state.recommendation
#     recommended_options = state.recommendation_options
    

#     print("List of Recommendations in task_execution_agent: ", recommended_output)
#     if "No action needed" not in recommended_output:

#         chosen_recommendation = send_notification("Recommendations by EMOFI", recommended_output,recommended_options)
#         print("Chosen recommendation: ", chosen_recommendation)
#         if chosen_recommendation:
#             print("Opening recommendations...")
#             is_opened = open_recommendations(chosen_recommendation)
#             state.executed = True
#             return {
#                     "executed": True,
#                 }



def task_execution_agent(state):
    recommended_output = state.recommendation
    recommended_options = state.recommendation_options
    
    # Ensure we have recommendations to process
    if not recommended_output or "No action needed" in recommended_output:
        return {"executed": False}
    if "No action needed" not in recommended_output:
        app_state.executed = True
        # pickle_save()
        # print("Task executed: ", app_state.executed)

        # while pickle_load().executedApp == False:
        #     print("waiting for reply..")
        #     time.sleep(2)

        # selectedRecommendation = pickle_load().selectedRecommendation
        # selectedApp = pickle_load().selectedApp

        # chosen_recommendation = {}

        # for i in app_state.recommendations:
        #     if(i['recommendation'] == selectedRecommendation):
        #         for j in i['recommendation_options']:
        #             if(j['app_name'] == selectedApp):
        #                 chosen_recommendation = j
        #                 break
        #         break

        # print("Executed task with recommendation: ", chosen_recommendation)

        # app_state.reset()
        # pickle_save()
    
        
    chosen_recommendation = send_notification(
        "Recommendations by EMOFI", 
        recommended_output,
        recommended_options
    )
    
    # Handle case where user didn't select anything
    if not chosen_recommendation:
        return {"executed": False}
        
    # Get open results safely
    result = open_recommendations(chosen_recommendation)
    if not result:
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
            "wait_start_time": time.time()
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
    # Reset tracking flags
    return {
        "executed": False,
        "action_time_start": None,
        "open_app_handle": None,
        "app_type": None,
        "continue_waiting": None
    }


# def recommendation_agent(state):
#     # Early exit if no task detected
#     if "No Need to Detect Task" in state.detected_task or not state.detected_task:
#         print("[Agent] No task detected, skipping recommendation.")
#         return {"recommendation": "No action needed"}
    
#     emotion = state.average_emotion.lower()
#     detected_task = state.detected_task.lower() if state.detected_task else "unknown"
#     print(f"[Agent] Calculating recommendation for emotion: {emotion} and task: {detected_task}")
    
#     negative_emotions = ["angry", "sad", "fear", "disgust", "stress", "boring"]
    
#     # Exit if not negative emotion
#     if emotion not in negative_emotions:
#         return {"recommendation": "No action needed"}

#     prompt = f"""
#         User is feeling {emotion} and is currently working on: {detected_task}.
#         Suggest one concrete action to improve mood from this list. Priority order:
#         1. Play music
#         2. Watch funny videos
#         3. Take a break
#         4. Quick game
#         5. Only if user is coding and needs help: "Coding Bot"
#         6. Nothing

#         Respond ONLY with the exact phrase from the list.
#     """
#     try:
#         response = requests.post(
#             "https://087f647be26e.ngrok-free.app/api/generate",  # Use local endpoint
#             headers={"Content-Type": "application/json"},
#             json={
#                 "model": "qwen3:4b",
#                 "prompt": prompt,
#                 "stream": False,
#                 "options": {"temperature": 0.2},
                
#             }
#         )
        
#         # Handle HTTP errors
#         if response.status_code != 200:
#             print(f"API error ({response.status_code}): {response.text[:100]}...")
#             return {"recommendation": "No action needed"}
            
#         response_data = response.json()
#         print("Response from Ollama:", response_data)
#         # Clean response from <think> tags if present
#         recommendation = clean_think_tags(response_data.get('response', '')).strip()
        
#         # Validate response format
#         valid_actions = [
#             "Play music", "Watch funny videos", "Take a break", 
#             "Quick game", "Coding Bot", "Nothing"
#         ]
        
#         if recommendation not in valid_actions:
#             print(f"[Warning] Invalid recommendation: {recommendation}")
#             recommendation = "No action needed"
        
#         # Store in state
#         state.recommendation = recommendation
#         print(f"Recommendation: {recommendation}")
#         return {"recommendation": recommendation}
        
#     except Exception as e:
#         print(f"Error generating recommendation: {str(e)}")
#         return {"recommendation": "No action needed"}



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