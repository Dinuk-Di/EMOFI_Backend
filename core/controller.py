# import threading
# import time
# import cv2
# import torch
# import customtkinter as ctk
# from queue import Queue, Empty
# from collections import Counter, deque
# import numpy as np
# import winsound
# import traceback
# import os

# from winotify import Notification, audio

# from core.human_detector import human_present
# from core.emotion_detector import get_emotion
# from core.sleepy_detector import check_sleepy
# from core.hand_movement import detect_hand
# from core.agent_system import run_agent_system
# from core.voice_assistant import VoiceAssistant
# from database.db import get_connection, get_app_setting



import threading
import time
import cv2
import torch
import customtkinter as ctk
from queue import Queue, Empty
from collections import Counter, deque
import numpy as np
import winsound
import traceback
import os
import requests
from concurrent.futures import ThreadPoolExecutor, wait
from winotify import Notification, audio
from core.human_detector import human_present
from core.emotion_detector import get_emotion
# from core.sleepy_detector import check_sleepy
from core.hand_movement import detect_hand
from core.agent_system import run_agent_system
from core.voice_assistant import VoiceAssistant
from database.db import get_connection, get_app_setting


class FrameReader(threading.Thread):
    def __init__(self, frame_queue):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.running = True
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[CRITICAL] Failed to initialize camera.")
            return

        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.qsize() > 1:
                    self.frame_queue.get()
                self.frame_queue.put(frame)
            time.sleep(0.03)

        self.cap.release()

    def stop(self):
        self.running = False


class AppController:
    def __init__(self, log_queue=None):
        self.log_queue = log_queue
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue()
        self.running = False
        self.reader_thread = FrameReader(self.frame_queue)
        self.main_thread = None
        self.focus_time = get_app_setting("resetTime", 25)
        self.notify_time = get_app_setting("recommendationTime", 30)
        self.need_focus_mode = get_app_setting("focusDetection", 1)
        self.need_hand_mode = get_app_setting("handDetection", 1)
        self.last_seen = time.time()
        # self.eye_closed_since = None
        # self.alert_triggered = False
        # self.sleepy_mode = False
        self.agent_mode = False
        # self.sleepy_pause_until = 0

        self.data_buffer = deque(maxlen=120)
        self.emotion_log = []
        self.hand_log = []
        self.lock = threading.Lock()
        self.gpu_lock = threading.Lock()

        self.frame_count = 0
        self.window_start_time = time.time()
        self.emotion_counter = Counter()
        self.hand_counter = Counter()

        # Target processing FPS, tweak for balance between CPU load and responsiveness
        self.process_fps = 30 if torch.cuda.is_available() else 4
        self.frame_interval = 1.0 / self.process_fps
        self.last_process_time = 0

        # Read config flags once to avoid repeated checks
        self.focus_enabled = (self.need_focus_mode == 1)
        self.hand_enabled = (self.need_hand_mode == 1)

        # Persistent thread pool executor for heavy detection
        self.executor = ThreadPoolExecutor(max_workers=2)

        self._warmed = False

    def log(self, message):
        if self.log_queue:
            self.log_queue.put(message)
        print(message)

    def start(self):
        self.running = True
        self.reader_thread.start()
        self.main_thread = threading.Thread(target=self.run, daemon=True)
        self.main_thread.start()

    def stop(self):
        self.running = False
        self.reader_thread.stop()
        self.executor.shutdown(wait=False)

    def exercise(self):
        """Called when the Exercise button is clicked in the Dashboard."""
        self.log("[UI] Exercise button clicked. Sending signal to Pi...")
        
        # Run network request in a separate thread to prevent freezing the GUI
        threading.Thread(target=self._send_exercise_signal, daemon=True).start()

    def _send_exercise_signal(self):
        """Internal method to send the HTTP request."""
        # Raspberry Pi Configuration
        RPI_IP = '10.50.228.36' 
        PORT = 5000
        ENDPOINT = "/api/exercise" # Or specific endpoint like /api/exercise
        
        url = f"http://{RPI_IP}:{PORT}{ENDPOINT}"

        try:
            # Sending an empty JSON object as requested: json={}
            response = requests.post(url, json={}, timeout=2.0)
            
            if response.status_code == 200:
                self.log(f"[Exercise] Success! Signal sent to {url}")
            else:
                self.log(f"[Exercise] Failed. Pi responded: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            self.log(f"[Exercise] Error: Could not connect to {RPI_IP}")
        except Exception as e:
            self.log(f"[Exercise] Error: {e}")

    
        

    # def _notify_async(self):
    #     def _do_notify():
    #         try:
    #             icon_path = os.path.join(
    #                 os.path.dirname(__file__), "..", "assets", "res", "Icon.ico"
    #             )
    #             icon_path = os.path.abspath(icon_path) if os.path.exists(icon_path) else None

    #             toast = Notification(
    #                 app_id="EMOFI",
    #                 title="Focus Alert",
    #                 msg="You are not focused!",
    #                 icon=icon_path
    #             )
    #             toast.set_audio(audio.Mail, loop=True)
    #             toast.show()
    #         except Exception as e:
    #             self.log(f"[ERROR] Notification failed: {e}")
    #         try:
    #             winsound.MessageBeep()
    #         except Exception:
    #             pass

    #     threading.Thread(target=_do_notify, daemon=True).start()

    # def buzzer_and_notify(self):
    #     self.sleepy_mode = False
    #     self.eye_closed_since = None
    #     self.alert_triggered = False
    #     self.sleepy_pause_until = time.time() + 5
    #     self.data_buffer.clear()
    #     self._notify_async()


    def _warmup_models(self):
        if self._warmed:
            return
        self._warmed = True
        dummy = None
        try:
            dummy = self.frame_queue.get_nowait()
        except Exception:
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)

        small = cv2.resize(dummy, (224, 224))
        # We don't convert color here because get_emotion/detect_hand should handle input format or you can add conversion if needed

        def _warm_emotion():
            try:
                with torch.inference_mode():
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        _ = get_emotion(small)
            except Exception:
                pass

        def _warm_hand():
            try:
                with torch.inference_mode():
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        _ = detect_hand(small)
            except Exception:
                pass

        tasks = [self.executor.submit(_warm_emotion)]
        if self.hand_enabled:
            tasks.append(self.executor.submit(_warm_hand))

        wait(tasks, timeout=0.5)

    def _run_emotion(self, frame):
        try:
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    return get_emotion(frame)
        except Exception as e:
            self.log(f"[ERROR] Emotion: {e}")
            return []

    def _run_hand(self, frame):
        try:
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    return detect_hand(frame)
        except Exception as e:
            self.log(f"[ERROR] Hand: {e}")
            return []

    def run_agent_workflow(self):
        with self.lock:
            self.log("[AGENT] Starting agent system...")
            combined_log = self.get_combined_log()
            self.log(f"[AGENT] Processing {len(combined_log)} events: {combined_log}")
            try:
                run_agent_system(combined_log)
            except Exception as e:
                self.log(f"[AGENT ERROR] {e}")
                traceback.print_exc()
            finally:
                self.agent_mode = False
                self.data_buffer.clear()
                self.emotion_log.clear()
                self.hand_log.clear()
                self.emotion_counter.clear()
                self.hand_counter.clear()
                self.window_start_time = time.time()
                self.frame_count = 0
                self.log("[AGENT] Finished")

    def get_combined_log(self):
        total_emotions = len(self.emotion_log)
        total_hands = len(self.hand_log)

        if total_emotions == 0 and total_hands == 0:
            return []

        n_emotions = max(1, int(total_emotions * 0.7))
        n_hands = max(1, int(total_hands * 0.3))

        recent_emotions = list(self.emotion_log)[-n_emotions:]
        recent_hands = list(self.hand_log)[-n_hands:]

        return recent_emotions + recent_hands
    
    # Series running
    def run(self):
        self.log(f"[INFO] GPU Available: {torch.cuda.is_available()}")
        self._warmup_models()

        try:
            while self.running:
                try:
                    frame = self.frame_queue.get(timeout=1)
                except Empty:
                    continue

                if frame is None or frame.size == 0:
                    self.log("[WARN] Empty frame.")
                    continue

                now = time.time()
                if self.agent_mode:
                    time.sleep(0.01)
                    continue

                # Detect if a human is present
                try:
                    detected = human_present(frame)
                except Exception as e:
                    self.log(f"[ERROR] Human detection: {e}")
                    continue

                if not detected:
                    if now - self.last_seen >= self.focus_time:
                        self.log(f"[WARN] No human detected for {self.focus_time}s.")
                    continue

                self.last_seen = now

                # Sleepy state check
                # if self.focus_enabled:
                #     try:
                #         eye_closed = check_sleepy(frame)
                #         if eye_closed:
                #             if self.eye_closed_since is None:
                #                 self.eye_closed_since = now
                #             elif (now - self.eye_closed_since >= self.focus_time) and not self.alert_triggered:
                #                 self.alert_triggered = True
                #                 self.buzzer_and_notify()
                #         else:
                #             self.eye_closed_since = None
                #             self.alert_triggered = False
                #     except Exception as e:
                #         self.log(f"[ERROR] Sleepy detection: {e}")

                # # Skip computation during alert cooldown
                # if time.time() < self.sleepy_pause_until:
                #     self.sleepy_mode = True
                #     self.window_start_time = time.time()
                #     continue

                # Preprocess for inference
                proc_frame = cv2.resize(frame, (224, 224))
                proc_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)

                emotion_result = []
                hand_result = []

                # --- Sequential Execution Starts Here ---
                # Run emotion model first
                try:
                    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        emotion_result = get_emotion(proc_frame)
                except Exception as e:
                    self.log(f"[ERROR] Emotion detection: {e}")

                # Only run hand detection if emotion not found
                if not emotion_result:
                    try:
                        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                            hand_result = detect_hand(proc_frame)
                    except Exception as e:
                        self.log(f"[ERROR] Hand detection: {e}")

                # Log detections
                if emotion_result:
                    self.emotion_log.extend(emotion_result)
                    self.emotion_counter.update(emotion_result)
                    self.log(f"[Emotion Detection] Detected: {emotion_result}")

                if hand_result:
                    self.hand_log.extend(hand_result)
                    self.hand_counter.update(hand_result)
                    self.log(f"[Hand Detection] Detected: {hand_result}")

                # Trigger periodic agent workflow
                if (now - self.window_start_time >= self.notify_time and 
                    not self.agent_mode and len(self.emotion_log) > 0):
                    self.agent_mode = True
                    threading.Thread(target=self.run_agent_workflow, daemon=True).start()

        finally:
            self.reader_thread.stop()
            self.executor.shutdown(wait=False)
            self.log("[INFO] AppController stopped.")


    # Parallel running
    # def run(self):
    #     self.log(f"[INFO] GPU Available: {torch.cuda.is_available()}")

    #     self._warmup_models()

    #     try:
    #         while self.running:
    #             try:
    #                 frame = self.frame_queue.get(timeout=1)
    #             except Empty:
    #                 continue

    #             now = time.time()

    #             if frame is None or frame.size == 0:
    #                 self.log("[WARN] Empty frame.")
    #                 time.sleep(0.01)
    #                 continue

    #             if self.agent_mode:
    #                 time.sleep(0.01)
    #                 continue

    #             # Human detection
    #             try:
    #                 detected = human_present(frame)
    #             except Exception as e:
    #                 self.log(f"[ERROR] Human detection: {e}")
    #                 continue

    #             if not detected:
    #                 if now - self.last_seen >= self.focus_time:
    #                     self.log(f"[WARN] No human detected for {self.focus_time}s.")
    #                 time.sleep(0.01)
    #                 continue

    #             self.last_seen = now

    #             # # Sleepy detection (inline)
    #             # if self.focus_enabled:
    #             #     try:
    #             #         eye_closed = check_sleepy(frame)
    #             #         if eye_closed:
    #             #             if self.eye_closed_since is None:
    #             #                 self.eye_closed_since = now
    #             #             elif (now - self.eye_closed_since >= self.focus_time) and not self.alert_triggered:
    #             #                 self.alert_triggered = True
    #             #                 self.buzzer_and_notify()
    #             #         else:
    #             #             self.eye_closed_since = None
    #             #             self.alert_triggered = False
    #             #     except Exception as e:
    #             #         self.log(f"[ERROR] Sleepy detection: {e}")

    #             # # Sleepy pause active? Skip heavy detection
    #             # if time.time() < self.sleepy_pause_until:
    #             #     self.sleepy_mode = True
    #             #     self.window_start_time = time.time()
    #             #     continue

    #             # Preprocess frame for heavy models
    #             proc_frame = cv2.resize(frame, (224, 224))
    #             proc_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)

    #             # Submit heavy detectors asynchronously
    #             futures = {}
    #             futures['emotion'] = self.executor.submit(self._run_emotion, proc_frame)
    #             if self.hand_enabled:
    #                 futures['hand'] = self.executor.submit(self._run_hand, proc_frame)

    #             wait(list(futures.values()))

    #             emotion_result = futures['emotion'].result() or []
    #             hand_result = futures['hand'].result() if 'hand' in futures else []

    #             if emotion_result:
    #                 self.emotion_log.extend(emotion_result)
    #                 self.emotion_counter.update(emotion_result)
    #                 self.log(f"[Emotion Detection] Detected: {emotion_result}")

    #             if hand_result:
    #                 self.hand_log.extend(hand_result)
    #                 self.hand_counter.update(hand_result)
    #                 self.log(f"[Hand Detection] Detected: {hand_result}")

    #             # Trigger agent workflow periodically
    #             if now - self.window_start_time >= self.notify_time and not self.agent_mode and len(self.emotion_log) > 0:
    #                 self.agent_mode = True
    #                 threading.Thread(target=self.run_agent_workflow, daemon=True).start()

    #     finally:
    #         self.reader_thread.stop()
    #         self.executor.shutdown(wait=False)
    #         self.log("[INFO] AppController stopped.")







# class FrameReader(threading.Thread):
#     def __init__(self, frame_queue):
#         super().__init__(daemon=True)
#         self.frame_queue = frame_queue
#         self.running = True
#         self.cap = None
        

#     def run(self):
#         self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#         if not self.cap.isOpened():
#             self.cap = cv2.VideoCapture(0)
#         if not self.cap.isOpened():
#             print("[CRITICAL] Failed to initialize camera.")
#             return

#         while self.running:
#             ret, frame = self.cap.read()
#             if ret:
#                 if self.frame_queue.qsize() > 1:
#                     self.frame_queue.get()
#                 self.frame_queue.put(frame)
#             time.sleep(0.03)

#         self.cap.release()

#     def stop(self):
#         self.running = False


# class AppController:
#     def __init__(self, log_queue=None):
#         self.log_queue = log_queue
#         self.frame_queue = Queue(maxsize=2)
#         self.result_queue = Queue()
#         self.running = False
#         self.reader_thread = FrameReader(self.frame_queue)
#         self.main_thread = None
#         self.focus_time = get_app_setting("resetTime", 25)
#         self.notify_time = get_app_setting("recommendationTime", 30)
#         self.need_focus_mode = get_app_setting("focusDetection",1)
#         self.need_hand_mode = get_app_setting("handDetection",1)
#         self.last_seen = time.time()
#         self.eye_closed_since = None
#         self.alert_triggered = False
#         self.sleepy_mode = False
#         self.agent_mode = False
#         self.sleepy_pause_until = 0

#         self.data_buffer = deque(maxlen=120)    
#         self.emotion_log = []
#         self.hand_log = []
#         self.lock = threading.Lock()
#         self.gpu_lock = threading.Lock()

#         self.frame_count = 0
#         self.window_start_time = time.time()
#         self.emotion_counter = Counter()
#         self.hand_counter = Counter()
        
#         # self.voice_assistant = VoiceAssistant(self)
#         # self.voice_assistant.start()

#     def log(self, message):
#         if self.log_queue:
#             self.log_queue.put(message)
#         print(message)

#     def start(self):
#         self.running = True
#         self.reader_thread.start()
#         self.main_thread = threading.Thread(target=self.run, daemon=True)
#         self.main_thread.start()

#     def stop(self):
#         self.running = False
#         self.reader_thread.stop()

#     def buzzer_and_notify(self):
#         try:
#             icon_path = os.path.join(
#                 os.path.dirname(__file__), "..", "assets", "res", "Icon.ico"
#             )
#             icon_path = os.path.abspath(icon_path) if os.path.exists(icon_path) else None

#             toast = Notification(
#                 app_id="EMOFI",
#                 title="Focus Alert",
#                 msg="You are not focused!",
#                 icon=icon_path
#             )
#             toast.set_audio(audio.Default, loop=False)
#             toast.show()
#         except Exception as e:
#             self.log(f"[ERROR] Notification failed: {e}")

#         winsound.Beep(1000, 1000)
#         self.sleepy_mode = False
#         self.eye_closed_since = None
#         self.alert_triggered = False
#         self.sleepy_pause_until = time.time() + 5
#         self.data_buffer.clear()

#     def run_agent_workflow(self):
#         with self.lock:
#             self.log("[AGENT] Starting agent system...")
#             combined_log = self.get_combined_log()
#             self.log(f"[AGENT] Processing {len(combined_log)} events: {combined_log}")
#             try:
#                 run_agent_system(combined_log)
#             except Exception as e:
#                 self.log(f"[AGENT ERROR] {e}")
#                 traceback.print_exc()
#             finally:
                
#                 self.agent_mode = False
#                 self.data_buffer.clear()
#                 self.emotion_log.clear()
#                 self.hand_log.clear()
#                 self.emotion_counter.clear()
#                 self.hand_counter.clear()
#                 self.window_start_time = time.time()
#                 self.frame_count = 0
#                 self.log("[AGENT] Finished")

#     def get_combined_log(self):
#         total_emotions = len(self.emotion_log)
#         total_hands = len(self.hand_log)

#         if total_emotions == 0 and total_hands == 0:
#             return []

#         n_emotions = max(1, int(total_emotions * 0.7))
#         n_hands = max(1, int(total_hands * 0.3))

#         recent_emotions = list(self.emotion_log)[-n_emotions:]
#         recent_hands = list(self.hand_log)[-n_hands:]

#         return recent_emotions + recent_hands
    
#     def run(self):
#         self.log(f"[INFO] GPU Available: {torch.cuda.is_available()}")

#         # ✅ Check feature flags ONCE
#         focus_enabled = (self.need_focus_mode == 1)
#         hand_enabled = (self.need_hand_mode == 1)

#         # ✅ Prebuild which heavy detection threads to run
#         def detect_emotion(frame):
#             try:
#                 with self.gpu_lock:
#                     return get_emotion(frame)
#             except Exception as e:
#                 self.log(f"[ERROR] Emotion: {e}")
#                 return []

#         def detect_hand_model(frame):
#             try:
#                 return detect_hand(frame)
#             except Exception as e:
#                 self.log(f"[ERROR] Hand: {e}")
#                 return []

#         heavy_detectors = [("emotion", detect_emotion)]
#         if hand_enabled:
#             heavy_detectors.append(("hand", detect_hand_model))

#         try:
#             while self.running:
#                 try:
#                     frame = self.frame_queue.get(timeout=1)
#                 except Empty:
#                     continue

#                 current_time = time.time()
#                 self.frame_count += 1

#                 if frame is None or frame.size == 0:
#                     self.log("[WARN] Empty frame.")
#                     time.sleep(1)
#                     continue

#                 if self.agent_mode:
#                     time.sleep(1)
#                     continue

#                 # Human detection
#                 try:
#                     detected = human_present(frame)
#                     self.log(f"[Human Detection] Present: {detected}")
#                 except Exception as e:
#                     self.log(f"[ERROR] Human detection: {e}")
#                     traceback.print_exc()
#                     continue

#                 if not detected:
#                     if current_time - self.last_seen >= self.focus_time:
#                         self.log(f"[WARN] No human detected for {self.focus_time}s.")
#                     time.sleep(1)
#                     continue

#                 self.last_seen = current_time

#                 # ✅ Run sleepy detection inline if enabled
#                 if focus_enabled:
#                     try:
#                         eye_closed = check_sleepy(frame)
#                         self.log(f"[Sleepy] Eye closed: {eye_closed}")
#                         if eye_closed:
#                             if self.eye_closed_since is None:
#                                 self.eye_closed_since = current_time
#                             elif (current_time - self.eye_closed_since >= self.focus_time) and not self.alert_triggered:
#                                 self.alert_triggered = True
#                                 self.buzzer_and_notify()
#                         else:
#                             self.eye_closed_since = None
#                             self.alert_triggered = False
#                     except Exception as e:
#                         self.log(f"[ERROR] Sleepy detection: {e}")
#                         traceback.print_exc()

#                 # Skip heavy detection if sleepy pause active
#                 if time.time() < self.sleepy_pause_until:
#                     self.log("[INFO] Sleepy pause active, skipping emotion/hand detection.")
#                     self.emotion_counter.clear()
#                     self.hand_counter.clear()
#                     self.data_buffer.clear()
#                     self.emotion_log.clear()
#                     self.hand_log.clear()
#                     self.sleepy_mode = True
#                     self.window_start_time = time.time()
#                     time.sleep(0.5)
#                     continue

#                 # ✅ Run heavy models in parallel threads
#                 emotion_result, hand_result = [], []
#                 threads = []

#                 for name, func in heavy_detectors:
#                     if name == "emotion":
#                         t = threading.Thread(target=lambda: emotion_result.extend(func(frame)))
#                     else:  # hand
#                         t = threading.Thread(target=lambda: hand_result.extend(func(frame)))
#                     threads.append(t)

#                 for t in threads:
#                     t.start()
#                 for t in threads:
#                     t.join()

#                 # Log results
#                 if emotion_result:
#                     self.log(f"[Emotion Detection] Detected: {emotion_result}")
#                     self.emotion_log.extend(emotion_result)

#                 if hand_result:
#                     self.log(f"[Hand Detection] Detected: {hand_result}")
#                     self.hand_log.extend(hand_result)

#                 self.emotion_counter.update(emotion_result)
#                 self.hand_counter.update(hand_result)

#                 # Agent workflow trigger
#                 if current_time - self.window_start_time >= self.notify_time and not self.agent_mode and len(self.emotion_log) > 0:
#                     self.agent_mode = True
#                     threading.Thread(target=self.run_agent_workflow, daemon=True).start()

#                 time.sleep(5)

#         finally:
#             self.reader_thread.stop()
#             self.log("[INFO] AppController stopped.")


    # def run(self):
    #     self.log(f"[INFO] GPU Available: {torch.cuda.is_available()}")
    #     try:
    #         while self.running:
    #             try:
    #                 frame = self.frame_queue.get(timeout=1)
    #             except Empty:
    #                 continue

    #             current_time = time.time()
    #             self.frame_count += 1

    #             if frame is None or frame.size == 0:
    #                 self.log("[WARN] Empty frame.")
    #                 time.sleep(1)
    #                 continue

    #             if self.agent_mode:
    #                 time.sleep(1)
    #                 continue

    #             try:
    #                 detected = human_present(frame)
    #                 self.log(f"[Human Detection] Present: {detected}")
    #             except Exception as e:
    #                 self.log(f"[ERROR] Human detection: {e}")
    #                 traceback.print_exc()
    #                 continue

    #             if not detected:
    #                 if current_time - self.last_seen >= self.focus_time:
    #                     self.log(f"[WARN] No human detected for {self.focus_time}s.")
    #                 time.sleep(1)
    #                 continue

    #             self.last_seen = current_time
    #             print("Need Focus Mode", self.need_focus_mode)
    #             if self.need_focus_mode == 1:
    #                 def sleepy_check():
    #                     try:
    #                         eye_closed = check_sleepy(frame)
    #                         self.log(f"[Sleepy] Eye closed: {eye_closed}")
    #                         if eye_closed:
    #                             if self.eye_closed_since is None:
    #                                 self.eye_closed_since = current_time
    #                             elif (current_time - self.eye_closed_since >= self.focus_time) and not self.alert_triggered:
    #                                 self.alert_triggered = True
    #                                 self.buzzer_and_notify()
    #                         else:
    #                             self.eye_closed_since = None
    #                             self.alert_triggered = False
    #                     except Exception as e:
    #                         self.log(f"[ERROR] Sleepy detection: {e}")
    #                         traceback.print_exc()

    #                 threading.Thread(target=sleepy_check).start()

    #             # 👇 Skip if currently in sleepy pause
    #             if time.time() < self.sleepy_pause_until:
    #                 self.log("[INFO] Sleepy pause active, skipping emotion/hand detection.")
    #                 self.emotion_counter.clear()
    #                 self.hand_counter.clear()
    #                 self.data_buffer.clear()
    #                 self.emotion_log.clear()
    #                 self.hand_log.clear()
    #                 self.sleepy_mode = True
    #                 self.window_start_time = time.time()
    #                 time.sleep(0.5)
    #                 continue

    #             emotion_result, hand_result = [], []
    #             print("Need Hand Mode", self.need_hand_mode)
    #             def detect_emotion():
    #                 try:
    #                     with self.gpu_lock:
    #                         emotion_result.extend(get_emotion(frame))
    #                 except Exception as e:
    #                     self.log(f"[ERROR] Emotion: {e}")

    #             threads = [threading.Thread(target=detect_emotion)]

    #             if self.need_hand_mode == 1:
    #                 def detect_hand_thread():
    #                     try:
    #                         hand_result.extend(detect_hand(frame))
    #                     except Exception as e:
    #                         self.log(f"[ERROR] Hand: {e}")
    #                 threads.append(threading.Thread(target=detect_hand_thread))

    #             for t in threads:
    #                 t.start()
    #             for t in threads:
    #                 t.join()


    #             if emotion_result:
    #                 self.log(f"[Emotion Detection] Detected: {emotion_result}")
    #                 self.emotion_log.extend(emotion_result)

    #             if hand_result:
    #                 self.log(f"[Hand Detection] Detected: {hand_result}")
    #                 self.hand_log.extend(hand_result)

    #             self.emotion_counter.update(emotion_result)
    #             self.hand_counter.update(hand_result)

    #             if current_time - self.window_start_time >= self.notify_time and not self.agent_mode:
    #                 self.agent_mode = True
    #                 threading.Thread(target=self.run_agent_workflow, daemon=True).start()

    #             time.sleep(5)
    #     finally:
    #         self.reader_thread.stop()
    #         self.log("[INFO] AppController stopped.")



                # def detect_emotion():
                #     try:
                #         with self.gpu_lock:
                #             emotion_result.extend(get_emotion(frame))
                #     except Exception as e:
                #         self.log(f"[ERROR] Emotion: {e}")
                # t1 = threading.Thread(target=detect_emotion)

                # def detect_hand_thread():
                #     try:
                #         hand_result.extend(detect_hand(frame))
                #     except Exception as e:
                #         self.log(f"[ERROR] Hand: {e}")

                # t1 = threading.Thread(target=detect_emotion)
                # t2 = threading.Thread(target=detect_hand_thread)
                # t1.start()
                # t2.start()
                # t1.join()
                # t2.join()