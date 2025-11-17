# from typing import Dict, List, Optional, Any
# import base64
# from collections import Counter
# import re
# import time
# from langgraph.graph import StateGraph, END
# import requests
# from utils.desktop import capture_desktop
# import ctypes
# from collections import Counter
# from typing import List, Optional
# from pydantic import BaseModel
# from utils.tools import open_recommendations
# from database.db import get_apps, get_connection,add_agent_recommendations
# from dotenv import load_dotenv
# import os
# from winotify import Notification
# import threading
# import ctypes
# from openai import OpenAI
# from pydantic import BaseModel, Field
# from old_utils.state import app_state, pickle_load, pickle_save
# import psutil
# from win11toast import toast

# load_dotenv()

# API_KEY = os.getenv("DEEPSEEK_API_KEY")
# QWEN_API_KEY = os.getenv("QWEN_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_KEY")

# def run_agent_system(emotions):
#     initial_state = AgentState(
#         emotions=emotions,
#         average_emotion=None,
#         continue_workflow=None,
#         recommendation=None,
#         recommendation_options= [],
#         executed=False,
#         action_executed=None,
#         action_time_start=0
#     )
#     agent_workflow = create_workflow()
    
#     # Increase recursion limit
#     config = {"recursion_limit": 100}  # Allow up to 100 steps
    
#     return agent_workflow.invoke(initial_state, config=config)

# class AppRecommendation(BaseModel):
#     app_name: str = Field(description="Name of recommended application")
#     app_url: str = Field(description="URL or local path of the application")
#     search_query: str = Field(description="Search query if web-based application")
#     is_local: bool = Field(default=False, description="Whether the app is a local executable")

# class RecommendationResponse(BaseModel):
#     recommendation: str = Field(description="4-word mood improvement suggestions")
#     recommendation_options: List[AppRecommendation] = Field(description="Two app recommendations")
# class RecommendationList(BaseModel):
#     listofRecommendations: List[RecommendationResponse] = Field(description="List of 3 recommendations with options")

# class AgentState(BaseModel):
#     emotions: List[str]
#     average_emotion: Optional[str]
#     continue_workflow: Optional[bool]
#     recommendation: Optional[List[str]]
#     recommendation_options: Optional[List[List[AppRecommendation]]]
#     executed: Optional[bool]
#     action_executed: Optional[str]
#     action_time_start: Optional[float]
#     open_app_handle: Optional[Any] = None
#     app_type: Optional[str] = None
#     continue_waiting: Optional[bool] = None
#     wait_start_time: Optional[float] = None  # Track when waiting began


# def create_workflow():
#     workflow = StateGraph(AgentState)
#     workflow.add_node("calculate_emotion", average_emotion_agent)
#     workflow.add_node("interrupt_check", interrupt_check_agent)
#     workflow.add_node("generate_recommendation", recommendation_agent)
#     workflow.add_node("execute_action", task_execution_agent)
#     workflow.add_node("wait_for_close", wait_for_close_agent)  # New node
#     workflow.add_node("exit_action", task_exit_agent)
    
#     workflow.set_entry_point("calculate_emotion")
#     workflow.add_edge("calculate_emotion", "interrupt_check")
#     workflow.add_conditional_edges(
#         "interrupt_check",
#         lambda state: "generate_recommendation" if state.continue_workflow else END,
#     )
#     workflow.add_edge("generate_recommendation", "execute_action")
#     workflow.add_conditional_edges(
#         "execute_action", 
#         lambda state: "wait_for_close" if state.executed else END,
#     )
#     workflow.add_conditional_edges(
#         "wait_for_close",
#         lambda state: "wait_for_close" if state.continue_waiting else "exit_action",
#     )
#     workflow.add_edge("exit_action", END)
#     return workflow.compile()

# def average_emotion_agent(state):
#     """Calculate most frequent emotion from AgentState model"""
#     if not state.emotions:
#         return {"average_emotion": "neutral"}
#     print(f"[Agent] Emotions: {state.emotions}")
#     counter = Counter(state.emotions)
#     most_common = counter.most_common(1)[0][0]
#     print(f"[Agent] Average emotion: {most_common}")

#     app_state.averageEmotion = most_common
#     pickle_save()

#     return {"average_emotion": most_common}

# # Remove reasoning tags from the response
# def clean_think_tags(text):
#     cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
#     return cleaned_text.strip()
    

# def show_notification_with_ok(title, message):
#     """Show Windows message box with OK/Cancel buttons"""
#     MB_OKCANCEL = 0x01
#     IDOK = 1
#     result = ctypes.windll.user32.MessageBoxW(0, message, title, MB_OKCANCEL)
#     return result == IDOK

# def send_notification(Title, message, timeout=20):
#     base_path = os.path.dirname(os.path.abspath(__file__))  # current script folder
#     icon_path = os.path.join(base_path, "assets", "res", "Icon1.jpg")

#     print(f"[notification] Icon path: {icon_path}")

#     event = threading.Event()
#     user_action = False
#     print(f"[notification] {Title}")

#     if not os.path.exists(icon_path):
#         print(f"Warning: Icon file not found at {icon_path}")
#         icon = None
#     else:
#         icon = {
#             'src': icon_path,
#             'placement': 'appLogoOverride'
#         }

#     def on_click(*args):
#         nonlocal user_action
#         user_action = True
#         event.set()

#     try:
#         toast(str(Title), str(message), icon=icon, app_id="EMOFI", on_click=on_click, button='Continue', duration='long')
#         print("notification sent")
#     except Exception as e:
#         print(f"Error sending notification: {e}")

#     # Wait for event or timeout seconds
#     event_set = event.wait(timeout)

#     # If event was not set, user_action remains False => dismissed or timed out
#     return user_action if event_set else False

# def interrupt_check_agent(state):
#     print("[Agent] Running interrupt_check_agent...")
#     emotion = state.average_emotion
#     negative_emotions = ["Angry", "Sad", "Fear", "Disgust", "Stress", "Boring"]
    
#     # Always reset continue_workflow to False
#     state.continue_workflow = False
    
#     if emotion in negative_emotions:
#         print(f"[Agent] Negative emotion detected: {emotion}")
#         # Show blocking message box
#         user_responded = send_notification(
#             "Your Emotion Is Not Good",
#             "Shall we give some suggestions to boost your mood?"
#         )
        
#         if user_responded:
#             print("[Agent] User accepted recommendations")
#             state.continue_workflow = True
#     else:
#         print(f"[Agent] Neutral/positive emotion: {emotion}")

#     return {"continue_workflow": state.continue_workflow}


# def parse_llm_response(text):
#     try:
#         # Clean <think> tags if present
#         text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

#         # Extract recommendation
#         rec_match = re.search(r'recommendation:\s*(.+)', text)
#         recommendation = rec_match.group(1).strip() if rec_match else None

#         # Extract recommendation_options (raw list inside [])
#         options_match = re.search(r'recommendation_options:\s*\[(.*)\]', text, re.DOTALL)
#         options_raw = options_match.group(1).strip() if options_match else ""

#         # Convert (app_name: 'X', ...) → {"app_name": "X", ...}
#         options = []
#         for option_text in re.findall(r'\((.*?)\)', options_raw, re.DOTALL):
#             entry = {}
#             for kv in option_text.split(','):
#                 key, val = kv.split(':', 1)
#                 entry[key.strip()] = val.strip().strip("'\"")
#             options.append(entry)

#         return recommendation, options

#     except Exception as e:
#         print("[Agent] Error parsing LLM response block:", e)
#         return None, []

# def extract_json_from_text(text):
#     try:
#         # Find JSON between ```json and ```
#         match = re.search(r'```json\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
#         if match:
#             return match.group(1)

#         # If no code block, try to parse whole text
#         if text.strip().startswith("{") or text.strip().startswith("["):
#             return text.strip()

#         raise ValueError("No valid JSON found in text.")
#     except Exception as e:
#         print("[Agent] JSON extraction failed:", e)
#         return None

# def send_blocking_message(title, message):
#     MB_OK = 0x0
#     ctypes.windll.user32.MessageBoxW(0, message, title, MB_OK)

# def wait_for_close_agent(state):
#     MAX_WAIT_SECONDS = 300  # 5 minute timeout
    
#     # Check if we should stop waiting
#     if not state.continue_waiting or not state.open_app_handle:
#         return {"continue_waiting": False}
    
#     # Check timeout
#     elapsed = time.time() - state.wait_start_time
#     if elapsed > MAX_WAIT_SECONDS:
#         print(f"[Agent] Wait timeout after {MAX_WAIT_SECONDS} seconds")
#         return {
#             "continue_waiting": False,
#             "open_app_handle": None,
#             "app_type": None
#         }
        
#     # Check if app is closed
#     app_closed = False
    
#     if state.app_type == 'local':
#         try:
#             process = psutil.Process(state.open_app_handle)
#             app_closed = not process.is_running()
#         except psutil.NoSuchProcess:
#             app_closed = True
            
#     elif state.app_type == 'web':
#         try:
#             # This will throw if browser closed
#             state.open_app_handle.current_url
#         except Exception:
#             app_closed = True
    
#     # Update waiting status
#     if app_closed:
#         print("[Agent] Detected app closure")
#         return {
#             "continue_waiting": False,
#             "open_app_handle": None,
#             "app_type": None
#         }
    
#     # Wait before checking again
#     time.sleep(5)  # Increased sleep to reduce recursion
#     print(f"[Agent] Still waiting for app to close ({int(elapsed)}s elapsed)")
#     return {"continue_waiting": True}


# def parse_llm_response(text):
#     try:
#         # Clean <think> tags if present
#         text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

#         # Extract recommendation
#         rec_match = re.search(r'recommendation:\s*(.+)', text)
#         recommendation = rec_match.group(1).strip() if rec_match else None

#         # Extract recommendation_options (raw list inside [])
#         options_match = re.search(r'recommendation_options:\s*\[(.*)\]', text, re.DOTALL)
#         options_raw = options_match.group(1).strip() if options_match else ""

#         # Convert (app_name: 'X', ...) → {"app_name": "X", ...}
#         options = []
#         for option_text in re.findall(r'\((.*?)\)', options_raw, re.DOTALL):
#             entry = {}
#             for kv in option_text.split(','):
#                 key, val = kv.split(':', 1)
#                 entry[key.strip()] = val.strip().strip("'\"")
#             options.append(entry)

#         return recommendation, options

#     except Exception as e:
#         print("[Agent] Error parsing LLM response block:", e)
#         return None, []

# def extract_json_from_text(text):
#     try:
#         # Find JSON between ```json and ```
#         match = re.search(r'```json\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
#         if match:
#             return match.group(1)

#         # If no code block, try to parse whole text
#         if text.strip().startswith("{") or text.strip().startswith("["):
#             return text.strip()

#         raise ValueError("No valid JSON found in text.")
#     except Exception as e:
#         print("[Agent] JSON extraction failed:", e)
#         return None
    
# def recommendation_agent(state):
#     # if not state.detected_task or "No Need to Detect Task" in state.detected_task:
#     #     print("[Agent] No task detected – skipping recommendation.")
#     #     return {"recommendation": ["No action needed"], "recommendation_options": []}


#     emotion = state.average_emotion
#     # task = state.detected_task
#     print(f"[Agent] Processing for emotion={emotion!r}")

#     negative_emotions = ["Angry", "Sad", "Fear", "Disgust", "Stress", "Boring"]
#     if emotion not in negative_emotions:
#         print("[Agent] Mood is fine – no recommendation.")
#         return {"recommendation": ["No action needed"], "recommendation_options": []}

#     conn = get_connection()
#     if not conn:
#         print("[Agent] DB connection failed – skip recommendation.")
#         return {"recommendation": ["No action needed"], "recommendation_options": []}

#     available_apps = get_apps(conn)
#     print("[Agent] Available apps:", available_apps)
#     # prompt = f"""
#     #         User feels {emotion} while working on: {task}.
#     #         Looking for 3 four-word mood-improvement suggestions.

#     #         Installed apps (category|name|path):
#     #         {available_apps!r}

#     #         Return ONLY valid JSON. No text, no notes. Output must be an array of 3 objects:
#     #         - recommendation: exactly four words
#     #         - recommendation_options: array of 2 items each with:
#     #             app_name (str),
#     #             app_url (URL or local path),
#     #             search_query (str, only for web apps),
#     #             is_local (bool)
#     #         Conditions to check seriously before returning:
#     #         - Each recommendation must have 2 app options.
#     #         - No duplicates, do not use webapp if local app is available in installed apps.
#     #         - All URLs must start with "https://" and if we can use a search query in the given app then append <search_query> token where suitable since each app search query is different from one another . 
#     #         - Each local app sets `is_local: true`.
#     #         """

#     prompt = f"""
#             You are a recommendation engine.

#             Context:
#             - User feels: "{emotion}"
#             - Available installed apps (format: category | name | path):
#             {available_apps!r}

#             Goal:
#             Generate EXACTLY 3 mood-improvement suggestions, each consisting of:
#             - recommendation: A phrase of exactly FOUR words.
#             - recommendation_options: An array of EXACTLY 2 options per recommendation. Each option must include:
#                 - app_name: (string)
#                 - app_url: (either a valid HTTPS URL for web apps OR local file path for installed apps)
#                 - search_query: (string, required only for web apps)
#                 - is_local: (true if app is installed locally, false if web)

#             STRICT RULES:
#             1. Output ONLY valid JSON — no extra text, no explanations, no markdown.
#             2. JSON format: An array of 3 objects with keys: recommendation, recommendation_options.
#             3. Each recommendation must have TWO different apps (no duplicates across or within).
#             4. Prefer local apps over web apps if available.
#             5. For web apps:
#             - All URLs must start with "https://".
#             - Use "<search_query>" placeholder in the app_url instead of inserting actual query.
#             - Example web apps are YouTube, Spotify, Online Game (https://poki.com/), MyFlixer (https://myflixerz.to/).
#             6. For local apps:
#             - Use given path as app_url and set is_local = true.
#             - search_query is empty
#             7. Don't use same app in multiple recommendations.
#             8. Each recommendation must be exactly 4 words, meaningful, and mood-impro
            

#             Example of expected structure (do NOT include this in response):
#             [
#             {{
#                 "recommendation": "Take a quick break",
#                 "recommendation_options": [
#                 {{
#                     "app_name": "Spotify",
#                     "app_url": "https://open.spotify.com/search/<search_query>",
#                     "search_query": "relaxing music",
#                     "is_local": false
#                 }},
#                 {{
#                     "app_name": "KMPlayer",
#                     "app_url": "C:\\\\Program Files\\\\KMPlayer 64X\\\\KMPlayer.exe",
#                     "search_query": "",
#                     "is_local": true
#                 }}
#                 ]
#             }}
#             ]

#             Now, produce the final JSON output:
#             """


#     full_schema = RecommendationList.model_json_schema()

#     try:
#         # res = requests.post(
#         #     url="https://openrouter.ai/api/v1/chat/completions",
#         #     headers={
#         #         "Authorization": f"Bearer {QWEN_API_KEY}",
#         #         "Content-Type": "application/json"
#         #     },
#         #     data=json.dumps({
#         #         "model": "deepseek/deepseek-r1-0528-qwen3-8b:free",
#         #         "messages": [
#         #             {"role": "system", "content": "You are an assistant. Output must be valid JSON only."},
#         #             {"role": "user", "content": prompt}
#         #         ],
#         #         "response_format": {
#         #             "type": "json_schema",
#         #             "json_schema": {
#         #                 "name": "recommendation_list",
#         #                 "strict": True,
#         #                 "schema": full_schema
#         #             }
#         #         },
#         #         "structured_outputs": True
#         #     }
#         # ))

#         schema = {
#             "type": "object",
#                     "properties": {
#                         "listofRecommendations": {
#                             "type": "array",
#                             "items": {
#                                 "type": "object",
#                                 "properties": {
#                                     "recommendation": {"type": "string"},
#                                     "recommendation_options": {
#                                         "type": "array",
#                                         "items": {
#                                             "type": "object",
#                                             "properties": {
#                                                 "app_name": {"type": "string"},
#                                                 "app_url": {"type": "string"},
#                                                 "search_query": {"type": "string"},
#                                                 "is_local": {"type": "boolean"}
#                                             }
#                                         }
#                                     }
#                                 }
#                             }
#                         }
#                     }
#             }

#         # res = requests.post(
#         #      "https://d53cb0fd37cb.ngrok-free.app/api/generate",  # Use local endpoint
#         #     headers={"Content-Type": "application/json"},
#         #     json={
#         #         "model": "qwen3:4b",
#         #         "prompt": prompt,
#         #         "stream": False,
#         #         "options": {"temperature": 0.2},
#         #         "format": schema    
#         #     }
#         # )


#         # print("[Agent] API response:", res.json())
#         # if res.status_code != 200:
#         #     print(f"[Agent] API returned status {res.status_code}: {res.text[:200]}")
#         #     return {"recommendation": ["No action needed"], "recommendation_options": []}

#         # # raw_content = res.json()["choices"][0]["message"]["content"]
#         # raw_content = res.json()["response"]
#         # print("Raw Response Content:", raw_content)

#         # try:
#         #     parsed_data = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
#         # except json.JSONDecodeError:
#         #     print("[Agent] Failed to decode JSON.")
#         #     return {"recommendation": ["No action needed"], "recommendation_options": []}

#         # if "listofRecommendations" not in parsed_data or not isinstance(parsed_data["listofRecommendations"], list):
#         #     print("[Agent] Parsed data is not a valid list of dicts.")
#         #     return {"recommendation": ["No action needed"], "recommendation_options": []}

#         # try:
#         #     recommendation_objects = [RecommendationResponse(**item) for item in parsed_data["listofRecommendations"]]
#         #     app_state.recommendations = parsed_data["listofRecommendations"]
#         # except Exception as e:
#         #     print("[Agent] Exception parsing recommendation objects:", e)
#         #     return {"recommendation": ["No action needed"], "recommendation_options": []}

#         # resp_data = RecommendationList(listofRecommendations=recommendation_objects)

#         client = OpenAI(api_key=OPENAI_API_KEY)
        
#         response = client.responses.parse(
#             model="gpt-4o-2024-08-06",
#             input=[
#                 {"role": "system", "content": "Give the proper structured output."},
#                 {
#                     "role": "user",
#                     "content": prompt,
#                 },
#             ],
#             text_format=RecommendationList,
#         )

#         print("[Agent] API response:", response.output_parsed)

#         resp_data = response.output_parsed

#         listData = convert_recommendations(resp_data)
#         print("List data: ", listData)

#         app_state.recommendations = listData

#         # Extract recommendations
#         recommendations_list = [rec.recommendation for rec in resp_data.listofRecommendations]
#         recommendation_options_list = [rec.recommendation_options for rec in resp_data.listofRecommendations]

#         print("Final Recommendations:", recommendations_list)
#         print("Options:", recommendation_options_list)        
        
#         # Update state
#         state.recommendation = recommendations_list
#         state.recommendation_options = recommendation_options_list

#         try:
#             for i, recommendation_type in enumerate(recommendations_list):
#                 for option in recommendation_options_list[i]:
#                     recommed_app = option.app_name
#                     app_url = option.app_url
#                     search_query = option.search_query
#                     is_local = option.is_local

#                     add_agent_recommendations(
#                         conn,
#                         1,
#                         recommendation_type,
#                         recommed_app,
#                         app_url,
#                         search_query,
#                         is_local
#                     )
#         except Exception as e:
#             print("[Agent] Error adding recommendations to DB:", e)


#         return {
#             "recommendation": recommendations_list,
#             "recommendation_options": recommendation_options_list
#         }

#     except Exception as e:
#         print("[Agent] Unexpected error:", e)
#         return {
#             "recommendation": ["No action needed"],
#             "recommendation_options": [],
#             "listofRecommendations": RecommendationList(listofRecommendations=[])
#         }

# def send_blocking_message(title, message):
#     MB_OK = 0x0
#     ctypes.windll.user32.MessageBoxW(0, message, title, MB_OK)

# def task_execution_agent(state):
#     recommended_output = state.recommendation
#     recommended_options = state.recommendation_options
    
#     print("List of Recommendations in task_execution_agent: ", recommended_output)
#     print("Recommendation Options in task_execution_agent: ", recommended_options)
#     if "No action needed" not in recommended_output:
#         app_state.executed = True
#         pickle_save()
#         print("Task executed: ", app_state.executed)

#         while pickle_load().executedApp == False:
#             print("waiting for reply..")
#             time.sleep(2)

#         selectedRecommendation = pickle_load().selectedRecommendation
#         selectedApp = pickle_load().selectedApp
#         selectedSearchQuery = pickle_load().searchQuery

#         chosen_recommendation = {}

#         for i in app_state.recommendations:
#             if(i['recommendation'] == selectedRecommendation):
#                 for j in i['recommendation_options']:
#                     if(j['app_name'] == selectedApp):
#                         chosen_recommendation = j
#                         chosen_recommendation['search_query'] = selectedSearchQuery
#                         break
#                 break

#         print("Executed task with recommendation: ", chosen_recommendation)

#         app_state.reset()
#         pickle_save()

#         #chosen_recommendation = send_notification("Recommendations by EMOFI", recommended_output,recommended_options)
#         print("Chosen recommendation: ", chosen_recommendation)
#         if chosen_recommendation:
#             print("Opening recommendations...")
#             open_recommendations(chosen_recommendation)
#             state.executed = True
#             return {
#                     "executed": True,
#                 }
                    
# def task_exit_agent(state):
#     task_executed = True
#     if not state.executed:
#         return {"executed": False, "action_time_start": None}
#     print("Thread is running")
#     while task_executed:
#         time.sleep(20)
#         task_executed = False
#     print("Thread is closed")
#     return {"executed": False, "action_time_start": None}

# def task_detection_agent(state):
#     try:
#         if state.average_emotion == "Neutral" or state.average_emotion == "Happy" or state.average_emotion == "Surprise":
#             print("[Agent] No task detection needed for neutral emotion.")
#             return {"detected_task": "No Need to Detect Task"}
#         # Capture screenshot as a base64 string (possibly with prefix)
#         screenshot = capture_desktop()
#         if not screenshot:
#             raise ValueError("Failed to capture screenshot")
#         # Remove data URI prefix if present
#         if screenshot.startswith('data:image'):
#             screenshot = screenshot.split(',')[1]

#         # Validate base64 string (optional, for debugging)
#         try:
#             base64.b64decode(screenshot)
#         except Exception as decode_err:
#             raise ValueError(f"Invalid base64 screenshot: {decode_err}")

#         # Send the raw base64 string (no prefix) to Ollama
#         # response = ollama.generate(
#         #     model="llava:7b",
#         #     prompt="Describe user's current activity. Focus on software and tasks.",
#         #     images=[screenshot]
#         # )
#         headers = {
#             "Connection": "close",  # Disable keep-alive
#             "Content-Type": "application/json"
#         }
#         response = requests.post(
#             "https://d53cb0fd37cb.ngrok-free.app/api/generate",
#             headers=headers,
#             json={
#                 "model": "llava:7b",
#                 "prompt": "Describe user's current activity. Focus on software and tasks.",
#                 "images": [screenshot],
#                 "stream": False
#             }
#         )

#         # Handle HTTP errors
#         if response.status_code != 200:
#             print(f"API error ({response.status_code}): {response.text[:100]}...")
#             return {"detected_task": "unknown"}

#         # Parse JSON response
#         response_data = response.json()
#         detected_task = response_data.get('response', '').strip()
#         state.detected_task = detected_task
#         print(f"Detected task: {detected_task}")
#         return {"detected_task": detected_task}

#     except Exception as e:
#         print(f"Error detecting task: {str(e)}")
#         return {"detected_task": "unknown"}
    
# def convert_recommendations(recommendations: RecommendationList):
#     converted = []
#     for rec in recommendations.listofRecommendations:
#         rec_dict = {
#             "recommendation": normalize_str(rec.recommendation),
#             "recommendation_options": []
#         }
#         for app in rec.recommendation_options:
#             rec_dict["recommendation_options"].append({
#                 "app_name": normalize_str(app.app_name),
#                 "app_url": normalize_str(app.app_url),
#                 "search_query": normalize_str(app.search_query),
#                 "isLocal": bool(app.is_local)
#             })
#         converted.append(rec_dict)
#     return converted

# def normalize_str(value) -> str:
#     if value is None:
#         return ""
#     if isinstance(value, list):
#         return " ".join(map(str, value))
#     return str(value)


import os
import json
import time
import ctypes
import numpy as np
import psutil
from collections import Counter
from typing import List, Optional, Any, Dict
from pydantic import BaseModel
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from openai import OpenAI
import winsound
from database.db import managed_connection, get_apps, add_agent_recommendations, add_rl_experience
from core.rl_model import load_bandit_model, get_context_features, get_action_features
from ui.notification import send_notification
from utils.tools import open_recommendations, close_app_by_pid, close_web_driver
from core.models import AppRecommendation, RecommendationList

# --- Load Environment ---
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ==============================================================================
# === 1. Agent State Model
# ==============================================================================

class AgentState(BaseModel):
    emotions: List[str]
    average_emotion: Optional[str] = None
    continue_workflow: Optional[bool] = None
    
    recommendation: Optional[List[str]] = None
    recommendation_options: Optional[List[List[AppRecommendation]]] = None
    
    rl_model: Optional[Any] = None
    context_features_dict: Optional[Dict[str, Any]] = None
    context_features_vec: Optional[np.ndarray] = None
    
    executed: Optional[bool] = False
    selected_action: Optional[AppRecommendation] = None
    action_features_dict: Optional[Dict[str, Any]] = None
    action_features_vec: Optional[np.ndarray] = None
    open_app_handle: Optional[Any] = None  # Will be PID (int) or Selenium Driver
    app_type: Optional[str] = None         # 'local' or 'web'
    
    continue_waiting: Optional[bool] = None
    wait_start_time: Optional[float] = None
    reward: Optional[float] = None
    
    class Config:
        arbitrary_types_allowed = True

# ==============================================================================
# === 2. Agent System Setup (LangGraph)
# ==============================================================================

def run_agent_system(emotions: List[str]):
    """Initializes and invokes the agent workflow."""
    initial_state = AgentState(emotions=emotions)
    agent_workflow = create_workflow()
    config = {"recursion_limit": 100}
    
    print("===== EMOFI Agent System Starting =====")
    final_state = agent_workflow.invoke(initial_state, config=config)
    print("===== EMOFI Agent System Finished =====")
    return final_state

def create_workflow() -> StateGraph:
    """Defines and compiles the agent's graph structure."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("load_model", load_model_agent)
    workflow.add_node("calculate_emotion", average_emotion_agent)
    workflow.add_node("interrupt_check", interrupt_check_agent)
    workflow.add_node("generate_recommendation", recommendation_agent)
    workflow.add_node("execute_action", task_execution_agent)
    workflow.add_node("wait_for_close", wait_for_close_agent)
    workflow.add_node("log_experience", log_experience_agent)
    workflow.add_node("exit_action", task_exit_agent)
    
    workflow.set_entry_point("load_model")
    workflow.add_edge("load_model", "calculate_emotion")
    workflow.add_edge("calculate_emotion", "interrupt_check")
    
    workflow.add_conditional_edges(
        "interrupt_check",
        lambda state: "generate_recommendation" if state.continue_workflow else "exit_action",
    )
    workflow.add_edge("generate_recommendation", "execute_action")
    workflow.add_conditional_edges(
        "execute_action", 
        lambda state: "wait_for_close" if state.executed else "exit_action",
    )
    workflow.add_conditional_edges(
        "wait_for_close",
        lambda state: "wait_for_close" if state.continue_waiting else "log_experience",
    )
    workflow.add_edge("log_experience", "exit_action")
    workflow.add_edge("exit_action", END)
    
    return workflow.compile()

# ==============================================================================
# === 3. Agent Nodes
# ==============================================================================

def load_model_agent(state: AgentState) -> Dict[str, Any]:
    """Loads the RL bandit model into the state."""
    model = load_bandit_model()
    return {"rl_model": model}

def average_emotion_agent(state: AgentState) -> Dict[str, Any]:
    """Calculate most frequent emotion and featurize the context."""
    if not state.emotions:
        avg_emotion = "neutral"
    else:
        avg_emotion = Counter(state.emotions).most_common(1)[0][0]
        
    print(f"[Agent] Average emotion: {avg_emotion}")
    
    emotion_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotion_map = {name.lower(): i for i, name in enumerate(emotion_list)}
    
    context_dict, context_vec = get_context_features()
    context_dict["emotion"] = avg_emotion.lower()
    
    emotion_index = emotion_map.get(avg_emotion.lower(), 6) # Default to Neutral
    context_vec[0, emotion_index] = 1.0
    
    return {
        "average_emotion": avg_emotion,
        "context_features_dict": context_dict,
        "context_features_vec": context_vec
    }

def show_windows_popup(title, message):
    """Show Windows message box with OK/Cancel buttons"""
    winsound.MessageBeep()
    MB_OKCANCEL = 0x01
    IDOK = 1
    # 0x1000 = MB_TOPMOST
    result = ctypes.windll.user32.MessageBoxW(0, message, title, MB_OKCANCEL | 0x1000) 
    return result == IDOK

def interrupt_check_agent(state: AgentState) -> Dict[str, Any]:
    """Asks user for permission to intervene if emotion is negative."""
    print("[Agent] Running interrupt_check_agent...")
    emotion = state.average_emotion
    negative_emotions = ["Angry", "Sad", "Fear", "Disgust", "Stress", "Boring"]
    
    continue_workflow = False
    
    if emotion in negative_emotions:
        print(f"[Agent] Negative emotion detected: {emotion}")
        user_responded = show_windows_popup(
            "Your Emotion Is Not Good",
            "Shall we give some suggestions to boost your mood?"
        )
        if user_responded:
            print("[Agent] User accepted recommendations")
            continue_workflow = True
        else:
            print("[Agent] User declined recommendations")
    else:
        print(f"[Agent] Neutral/positive emotion: {emotion}")

    return {"continue_workflow": continue_workflow}

def recommendation_agent(state: AgentState) -> Dict[str, Any]:
    """Generates candidates with LLM, then ranks them with the RL model."""
    emotion = state.average_emotion
    print(f"[Agent] Processing for emotion={emotion!r}")

    try:
        with managed_connection() as conn:
            available_apps = get_apps(conn) # Fetches (category, name, app_id, path)
    except Exception as e:
        print(f"[Agent] DB connection failed: {e}")
        return {"recommendation": ["No action needed"], "recommendation_options": []}

    # 1. --- Candidate Generation (LLM) ---
    prompt = f"""
        You are a recommendation engine.
        User feels: "{emotion}"
        Available installed apps (category | name | app_id | path): {available_apps!r}

        Goal:
        Generate EXACTLY 3 mood-improvement suggestions.
        Each suggestion must have:
        - recommendation: A phrase of exactly FOUR words.
        - recommendation_options: An array of EXACTLY 2 options per recommendation.

        Each option object MUST include:
        - app_name: (string)
        - app_url: (valid HTTPS URL or local file path from input)
        - search_query: (string, default to empty "" if not needed)
        - is_local: (true if local, false if web)
        - category: (string: must be one of {APP_CATEGORIES})

        STRICT RULES:
        1. Output ONLY valid JSON in the specified format.
        2. Each of the 3 recommendations must have 2 *different* app options.
        3. Prefer local apps from the list if they fit the mood.
        4. For local apps, use the 'path' as 'app_url' and set 'is_local' to true.
        5. For web apps:
           - URLs must start with "https://".
           - Use "<search_query>" placeholder in the app_url if search is relevant (e.g., YouTube).
        
        Produce the final JSON output:
        """

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "You are an assistant that only outputs valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"}
        )
        
        llm_json = json.loads(response.choices[0].message.content)
        llm_recommendations = RecommendationList.model_validate(llm_json)

    except Exception as e:
        print(f"[Agent] LLM API error: {e}")
        return {"recommendation": ["No action needed"], "recommendation_options": []}

    # 2. --- RL Ranking ---
    print("[RL] Ranking LLM candidates...")
    context_vec = state.context_features_vec
    rl_model = state.rl_model
    
    ranked_recommendations = []
    ranked_options = []

    for rec_response in llm_recommendations.listofRecommendations:
        candidates = rec_response.recommendation_options
        
        action_features_list = []
        for app in candidates:
            _, action_vec = get_action_features(app)
            action_features_list.append(action_vec)
        
        scores = rl_model.predict_score(context_vec, action_features_list)
        print(f"[RL] Scores for '{rec_response.recommendation}': {scores}")
        
        sorted_candidates = [
            app for _, app in sorted(
                zip(scores, candidates), 
                key=lambda pair: pair[0], 
                reverse=True
            )
        ]
        
        ranked_recommendations.append(rec_response.recommendation)
        ranked_options.append(sorted_candidates)

    # 3. --- Save and Return ---
    print("[Agent] Final Ranked Recommendations:", ranked_recommendations)

    try:
        with managed_connection() as conn:
            for i, recommendation_type in enumerate(ranked_recommendations):
                for option in ranked_options[i]:
                    add_agent_recommendations(
                        conn, 1, recommendation_type, # Assuming user_id=1
                        option.app_name, option.app_url,
                        option.search_query, option.is_local
                    )
    except Exception as e:
        print(f"[Agent] Error adding recommendations to DB: {e}")

    return {
        "recommendation": ranked_recommendations,
        "recommendation_options": ranked_options
    }

def task_execution_agent(state: AgentState) -> Dict[str, Any]:
    """Shows notification and executes the user's chosen action."""
    recommended_output = state.recommendation
    recommended_options = state.recommendation_options
    
    if not recommended_output or "No action needed" in recommended_output[0]:
        return {"executed": False}
    
    # send_notification is a blocking Tkinter call from ui/notification.py
    chosen_app_dict = send_notification(
        "Recommendations by EMOFI", 
        recommended_output,
        recommended_options
    )
    
    if not chosen_app_dict:
        print("[Agent] User declined recommendation.")
        return {"executed": False}
    
    # Convert dict back to AppRecommendation object
    chosen_recommendation = AppRecommendation.model_validate(chosen_app_dict)
    
    # open_recommendations is from utils/tools.py
    result = open_recommendations(chosen_recommendation)
    if not result:
        return {"executed": False}
        
    is_opened, app_handle, app_type = result
    
    if is_opened:
        action_dict, action_vec = get_action_features(chosen_recommendation)
        return {
            "executed": True,
            "selected_action": chosen_recommendation,
            "action_features_dict": action_dict,
            "action_features_vec": action_vec,
            "open_app_handle": app_handle,
            "app_type": app_type,
            "continue_waiting": True,
            "wait_start_time": time.time()
        }
    
    return {"executed": False}


def wait_for_close_agent(state: AgentState) -> Dict[str, Any]:
    """
    Waits for the recommended app to be closed OR times out.
    Calculates dwell time as reward.
    """
    MAX_WAIT_SECONDS = 300  # 5 minute timeout
    
    if not state.continue_waiting:
        return {"continue_waiting": False, "reward": 0.0}

    # Handle case where app was launched without a handle (e.g., webbrowser fallback)
    if not state.open_app_handle:
        print("[Agent] No app handle to track. Waiting for timeout.")
        # We can't detect close, so we just wait for the full timeout.
        if (time.time() - state.wait_start_time) > MAX_WAIT_SECONDS:
             print("[Agent] No-handle wait complete.")
             return {"continue_waiting": False, "reward": MAX_WAIT_SECONDS}
        
        time.sleep(5)
        return {"continue_waiting": True} # Continue loop

    elapsed = time.time() - state.wait_start_time
    
    # 1. Check for Timeout
    if elapsed > MAX_WAIT_SECONDS:
        print(f"[Agent] Wait timeout after {MAX_WAIT_SECONDS} seconds")
        if state.app_type == 'local':
            close_app_by_pid(state.open_app_handle)
        elif state.app_type == 'web':
            close_web_driver(state.open_app_handle)
            
        return {
            "continue_waiting": False,
            "reward": MAX_WAIT_SECONDS, # Cap reward at max wait
            "open_app_handle": None,
            "app_type": None
        }
        
    # 2. Check if App is Closed
    app_closed = False
    
    if state.app_type == 'local':
        try:
            app_closed = not psutil.Process(state.open_app_handle).is_running()
        except psutil.NoSuchProcess:
            app_closed = True
            
    elif state.app_type == 'web':
        try:
            _ = state.open_app_handle.title # Check if driver is reachable
        except Exception:
            app_closed = True
            
    if app_closed:
        print(f"[Agent] Detected app closure. Dwell time: {elapsed:.2f}s")
        if state.app_type == 'web':
             try: state.open_app_handle.quit() # Final cleanup
             except: pass
        
        return {
            "continue_waiting": False,
            "reward": elapsed,
            "open_app_handle": None,
            "app_type": None
        }
    
    # 3. Not closed and not timed out: Continue waiting
    time.sleep(5) 
    print(f"[Agent] Still waiting for app to close ({int(elapsed)}s elapsed)")
    return {"continue_waiting": True}


def log_experience_agent(state: AgentState) -> Dict[str, Any]:
    """Logs the (State, Action, Reward) tuple to the database."""
    # Only log meaningful interactions (e.g., > 1 second)
    if state.selected_action and state.reward is not None and state.reward > 1.0:
        print("[Agent] Logging RL experience to database...")
        
        context_dict = state.context_features_dict
        action_dict = state.action_features_dict
        reward = state.reward
        
        try:
            with managed_connection() as conn:
                add_rl_experience(
                    conn,
                    context_json=json.dumps(context_dict),
                    action_json=json.dumps(action_dict),
                    reward=reward
                )
            print("[Agent] Log successful.")
        except Exception as e:
            print(f"[Agent] Failed to log RL experience: {e}")
            
    elif state.reward is not None:
        print("[Agent] Interaction was too short, skipping RL log.")
        
    return {}

def task_exit_agent(state: AgentState) -> Dict[str, Any]:
    """Cleans up the state after a workflow run."""
    print("[Agent] Workflow finished. Resetting state.")
    return {
        "executed": False,
        "selected_action": None,
        "action_features_dict": None,
        "action_features_vec": None,
        "open_app_handle": None,
        "app_type": None,
        "continue_waiting": None,
        "wait_start_time": None,
        "reward": None
    }
