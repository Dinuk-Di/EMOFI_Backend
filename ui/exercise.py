import customtkinter as ctk
import cv2
from PIL import Image
from customtkinter import CTkImage
import threading
import time
import os

VIDEOS = [
    {"path": "assets/video/video1.mp4", "duration": 30},
    {"path": "assets/video/video2.mp4", "duration": 30},
    {"path": "assets/video/video3.mp4", "duration": 30},
]

REST_DURATION = 5

class ExerciseWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Exercise Player")
        self.root.geometry("600x500")
        self.root.resizable(False, False)

        self.is_running = False
        self.current_index = 0
        self.cap = None
        self.video_thread = None

        # Title
        ctk.CTkLabel(self.root, text="Workout Session", font=("Arial", 20, "bold")).pack(pady=10)

        # Video Frame
        self.video_frame = ctk.CTkLabel(self.root, text="")
        self.video_frame.pack(pady=10)

        # Timer Label
        self.timer_label = ctk.CTkLabel(self.root, text="00:00", font=("Arial", 30, "bold"))
        self.timer_label.pack(pady=10)

        # Buttons
        button_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        button_frame.pack(pady=20)

        self.start_btn = ctk.CTkButton(button_frame, text="Start", width=120, command=self.start_workout)
        self.start_btn.pack(side="left", padx=10)

        self.end_btn = ctk.CTkButton(button_frame, text="End", width=120, fg_color="red", hover_color="#aa0000",
                                     command=self.end_workout)
        self.end_btn.pack(side="left", padx=10)

    def start_workout(self):
        if self.is_running:
            return
        self.is_running = True
        self.current_index = 0
        self.video_thread = threading.Thread(target=self.play_videos)
        self.video_thread.start()

    def end_workout(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.timer_label.configure(text="00:00")
        self.video_frame.configure(image="", text="Workout Ended")

    def play_videos(self):
        while self.is_running and self.current_index < len(VIDEOS):
            video_info = VIDEOS[self.current_index]
            video_path = video_info["path"]

            if not os.path.exists(video_path):
                self.video_frame.configure(text=f"Video not found: {video_path}")
                return

            self.cap = cv2.VideoCapture(video_path)
            start_time = time.time()

            while self.is_running and (time.time() - start_time) < video_info["duration"]:
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((500, 300))
                imgtk = CTkImage(light_image=img, dark_image=img, size=(500, 300))
                self.video_frame.configure(image=imgtk)
                self.video_frame.image = imgtk

                remaining = int(video_info["duration"] - (time.time() - start_time))
                self.timer_label.configure(text=f"{remaining} sec")

                time.sleep(0.03)

            if self.cap:
                self.cap.release()

            if not self.is_running:
                break

            self.show_rest_screen()
            self.current_index += 1

        self.is_running = False
        self.timer_label.configure(text="Done!")
        self.video_frame.configure(image="", text="Workout Complete")

    def show_rest_screen(self):
        self.video_frame.configure(image="", text="Rest Time")
        for i in range(REST_DURATION, 0, -1):
            if not self.is_running:
                return
            self.timer_label.configure(text=f"Rest: {i} sec")
            time.sleep(1)


if __name__ == "__main__":
    root = ctk.CTk()
    app = ExerciseWindow(root)
    root.mainloop()
