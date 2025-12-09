import threading
import time
import cv2
import torch
import customtkinter as ctk
from queue import Queue, Empty
from collections import Counter, deque
import numpy as np
import traceback
import os
import requests
from concurrent.futures import ThreadPoolExecutor, wait
from winotify import Notification, audio
from core.human_detector import human_present
from core.emotion_detector import get_emotion
from core.hand_movement import detect_hand
from core.agent_system import run_agent_system
from database.db import get_app_setting

class FrameReader(threading.Thread):
    def __init__(self, frame_queue):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.running = True
        self.cap = None

    def run(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("[CRITICAL] Failed to initialize camera.")
            return

        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Keep queue fresh, drop old frames if queue is full
                if self.frame_queue.qsize() > 1:
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass
                self.frame_queue.put(frame)
            time.sleep(0.03)

        # Cleanup when loop ends
        if self.cap:
            self.cap.release()
        print("[INFO] Camera released.")

    def stop(self):
        self.running = False


class AppController:
    def __init__(self, log_queue=None):
        self.log_queue = log_queue
        self.frame_queue = Queue(maxsize=2)
        
        # --- State Flags ---
        self.running = False # Tracks if the system is currently active
        
        # --- Resources (Initialize as None) ---
        self.reader_thread = None
        self.main_thread = None
        self.executor = None
        
        # --- Settings ---
        self.focus_time = get_app_setting("resetTime", 25)
        self.notify_time = get_app_setting("recommendationTime", 30)
        self.need_focus_mode = get_app_setting("focusDetection", 1)
        self.need_hand_mode = get_app_setting("handDetection", 1)
        
        self.last_seen = time.time()
        self.agent_mode = False

        self.data_buffer = deque(maxlen=120)
        self.emotion_log = []
        self.hand_log = []
        self.lock = threading.Lock()

        self.frame_count = 0
        self.window_start_time = time.time()
        self.emotion_counter = Counter()
        self.hand_counter = Counter()

        self.process_fps = 30 if torch.cuda.is_available() else 4
        self.frame_interval = 1.0 / self.process_fps

        self.focus_enabled = (self.need_focus_mode == 1)
        self.hand_enabled = (self.need_hand_mode == 1)

        self._warmed = False

    def log(self, message):
        if self.log_queue:
            self.log_queue.put(message)
        print(message)

    def start(self):
        if self.running:
            self.log("[WARN] System is already running.")
            return

        self.log("[INFO] Starting Detection System...")
        self.running = True
        
        # 1. Reset Warmup flag so we re-warm the new executor threads
        self._warmed = False

        # 2. Clear the frame queue to remove old stale frames
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()

        # 3. Create NEW Executor instance
        self.executor = ThreadPoolExecutor(max_workers=2)

        # 4. Create NEW FrameReader thread instance
        self.reader_thread = FrameReader(self.frame_queue)
        self.reader_thread.start()

        # 5. Create NEW Main processing thread instance
        self.main_thread = threading.Thread(target=self.run, daemon=True)
        self.main_thread.start()
        
        self.log("[INFO] System Started.")

    def stop(self):
        if not self.running:
            return

        self.log("[INFO] Stopping System...")
        self.running = False
        
        # Stop the reader thread (closes camera)
        if self.reader_thread:
            self.reader_thread.stop()
            # We don't join() here to avoid freezing UI, let it finish naturally
        
        # Shutdown executor (frees up threads)
        if self.executor:
            self.executor.shutdown(wait=False)
            self.executor = None # Reset to None
        
        self.log("[INFO] System Stopped.")

    def exercise(self):
        self.log("[UI] Exercise button clicked. Sending signal to Pi...")
        threading.Thread(target=self._send_exercise_signal, daemon=True).start()

    def _send_exercise_signal(self):
        RPI_IP = '10.170.72.223' 
        PORT = 5000
        ENDPOINT = "/api/exercise"
        url = f"http://{RPI_IP}:{PORT}{ENDPOINT}"

        try:
            response = requests.post(url, json={}, timeout=2.0)
            if response.status_code == 200:
                self.log(f"[Exercise] Success: {url}")
            else:
                self.log(f"[Exercise] Failed. Pi responded: {response.status_code}")
        except requests.exceptions.ConnectionError:
            self.log(f"[Exercise] Error: Could not connect to {RPI_IP}")
        except Exception as e:
            self.log(f"[Exercise] ERROR — continuing loop: {e}")

    def _warmup_models(self):
        if self._warmed:
            return
        self._warmed = True
        self.log("[INFO] Warming up models...")
        
        dummy = None
        try:
            dummy = self.frame_queue.get(timeout=2) # Wait a bit for camera to produce frame
        except Empty:
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)

        small = cv2.resize(dummy, (224, 224))

        def _warm_emotion():
            try:
                with torch.inference_mode(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    _ = get_emotion(small)
            except Exception: pass

        def _warm_hand():
            try:
                with torch.inference_mode(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    _ = detect_hand(small)
            except Exception: pass

        tasks = [self.executor.submit(_warm_emotion)]
        if self.hand_enabled:
            tasks.append(self.executor.submit(_warm_hand))

        wait(tasks, timeout=2.0)
        self.log("[INFO] Warmup complete.")

    def run_agent_workflow(self):
        with self.lock:
            self.log("[AGENT] Starting agent system...")
            combined_log = self.get_combined_log()
            self.log(f"[AGENT] Processing {len(combined_log)} events.")
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
    
    def run(self):
        # This runs inside self.main_thread
        self.log(f"[INFO] GPU Available: {torch.cuda.is_available()}")
        
        # Important: Check running flag before warmup
        if not self.running: return
        self._warmup_models()

        try:
            while self.running:
                try:
                    frame = self.frame_queue.get(timeout=1)
                except Empty:
                    continue

                if frame is None or frame.size == 0:
                    continue

                now = time.time()
                
                # If agent is processing, skip frames to save resources
                if self.agent_mode:
                    time.sleep(0.01)
                    continue

                # --- Human Detection ---
                try:
                    detected = human_present(frame)
                except Exception as e:
                    self.log(f"[ERROR] Human detection: {e}")
                    continue

                if not detected:
                    if now - self.last_seen >= self.focus_time:
                        # Optional: Log only occasionally to avoid spam
                        pass 
                    continue

                self.last_seen = now

                # --- Inference Preprocessing ---
                proc_frame = cv2.resize(frame, (224, 224))
                proc_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)

                emotion_result = []
                hand_result = []

                # Ensure executor exists before submitting
                if not self.executor: break

                # --- Run Logic ---
                # 1. Emotion
                try:
                    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        emotion_result = get_emotion(proc_frame)
                except Exception as e:
                    self.log(f"[ERROR] Emotion: {e}")

                # 2. Hand (only if emotion empty)
                if not emotion_result:
                    try:
                        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                            hand_result = detect_hand(proc_frame)
                    except Exception as e:
                        self.log(f"[ERROR] Hand: {e}")

                # --- Logging ---
                if emotion_result:
                    self.emotion_log.extend(emotion_result)
                    self.emotion_counter.update(emotion_result)
                    self.log(f"[Emotion] {emotion_result}")

                if hand_result:
                    self.hand_log.extend(hand_result)
                    self.hand_counter.update(hand_result)
                    self.log(f"[Hand] {hand_result}")

                # --- Trigger Agent ---
                if (now - self.window_start_time >= self.notify_time and 
                    not self.agent_mode and len(self.emotion_log) > 0):
                    self.agent_mode = True
                    # Start agent thread
                    threading.Thread(target=self.run_agent_workflow, daemon=True).start()

        except Exception as e:
            self.log(f"[CRITICAL ERROR] Main Loop: {e}")
            traceback.print_exc()
        finally:
            # Ensure cleanup if the loop breaks unexpectedly
            if self.reader_thread and self.reader_thread.is_alive():
                self.reader_thread.stop()
            self.log("[INFO] Main processing thread finished.")