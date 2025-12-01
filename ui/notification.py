from typing import List
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import win32api
import win32con
import time
import subprocess
import webbrowser
from win11toast import toast
import os
import threading
icon_path = r'D:\RuhunaNew\Academic\Research\Facial_Recog_Repo\Group_50_Repo\DesktopApp\assets\res\Icon.jpg'
executer_path = r'executer.pyw'
import tkinter as tk
import os
import ctypes
import winsound

# def send_notification(title, recommendations, timeout=None):
#     user_choice = {"value": None}

#     def on_click(rec):
#         user_choice["value"] = rec
#         root.destroy()

#     # Create the main window
#     root = tk.Tk()
#     root.title(title)
#     root.overrideredirect(True)  # Remove window border
#     root.attributes("-topmost", True)
#     root.configure(bg="#2b2b2b")

#     # Set window size and position
#     width, height = 300, 200
#     x = root.winfo_screenwidth() - width - 20
#     y = 50
#     root.geometry(f"{width}x{height}+{x}+{y}")

#     # Frame for styling
#     frame = tk.Frame(root, bg="#2b2b2b", bd=2)
#     frame.place(relwidth=1, relheight=1)

#     # Title
#     tk.Label(frame, text=title, bg="#2b2b2b", fg="white", font=("Segoe UI", 12, "bold")).pack(pady=(10, 5))

#     # Instructions
#     tk.Label(frame, text="Choose an option:", bg="#2b2b2b", fg="white", font=("Segoe UI", 10)).pack(pady=(0, 10))

#     # Buttons
#     for rec in recommendations:
#         btn = tk.Button(frame, text=rec, width=25, bg="#3c3f41", fg="white", font=("Segoe UI", 9),
#                         relief="flat", highlightthickness=0, command=lambda r=rec: on_click(r))
#         btn.pack(pady=3)

#     # Optional timeout
#     if timeout:
#         root.after(timeout, root.destroy)

#     root.mainloop()
#     return user_choice["value"]


import tkinter as tk

def send_notification(title, recommended_output, recommended_options, timeout=None):
    winsound.MessageBeep()
    selected_app = {"value": None}

    def show_app_options(app_options):
        for widget in frame.winfo_children():
            widget.destroy()

        tk.Label(frame, text="Choose an app:", bg="#2b2b2b", fg="white", font=("Segoe UI", 10)).pack(pady=(0, 10))

        for option in app_options:
            btn = tk.Button(frame, text=option.app_name, width=25, bg="#3c3f41", fg="white", font=("Segoe UI", 9),
                            relief="flat", highlightthickness=0, command=lambda opt=option: handle_app_selection(opt))
            btn.pack(pady=3)

    def handle_app_selection(option):
        # If local app, select immediately
        if option.is_local:
            select_app(option, None)
        else:
            # Show input field for search query
            show_search_input(option)

    def show_search_input(option):
        for widget in frame.winfo_children():
            widget.destroy()

        tk.Label(frame, text=f"Enter search text for {option.app_name} (or skip):", bg="#2b2b2b", fg="white",
                 font=("Segoe UI", 10), wraplength=280).pack(pady=(10, 5))

        query_entry = tk.Entry(frame, width=25, font=("Segoe UI", 10))
        query_entry.pack(pady=10)

        btn_frame = tk.Frame(frame, bg="#2b2b2b")
        btn_frame.pack(pady=10)

        # Confirm with query
        tk.Button(btn_frame, text="Search", bg="#3c3f41", fg="white", width=10,
                  command=lambda: select_app(option, query_entry.get())).pack(side="left", padx=5)

        # Skip query
        tk.Button(btn_frame, text="Skip", bg="#3c3f41", fg="white", width=10,
                  command=lambda: select_app(option, None)).pack(side="left", padx=5)

    def select_app(option, custom_query):
        selected_app["value"] = {
            "app_name": option.app_name,
            "app_url": option.app_url,
            "search_query": custom_query if custom_query else option.search_query,
            "is_local": option.is_local,
            "category": option.category
        }
        root.destroy()

    def select_recommendation(index):
        show_app_options(recommended_options[index])

    # Create main window
    root = tk.Tk()
    root.title(title)
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg="#2b2b2b")

    width, height = 320, 250
    x = root.winfo_screenwidth() - width - 20
    y = 50
    root.geometry(f"{width}x{height}+{x}+{y}")

    frame = tk.Frame(root, bg="#2b2b2b", bd=2)
    frame.place(relwidth=1, relheight=1)

    tk.Label(frame, text=title, bg="#2b2b2b", fg="white", font=("Segoe UI", 12, "bold")).pack(pady=(10, 5))
    tk.Label(frame, text="Choose an option:", bg="#2b2b2b", fg="white", font=("Segoe UI", 10)).pack(pady=(0, 10))

    for idx, rec in enumerate(recommended_output):
        tk.Button(frame, text=rec, width=25, bg="#3c3f41", fg="white", font=("Segoe UI", 9),
                  relief="flat", highlightthickness=0, command=lambda i=idx: select_recommendation(i)).pack(pady=3)

    if timeout:
        root.after(timeout, root.destroy)

    root.mainloop()
    return selected_app["value"]



def execute_task(option):
    time.sleep(5)
    try:
        app_name = option.get("app_name")
        app_url = option.get("app_url")
        search_query = option.get("search_query")

        if app_url.startswith("http"):
            if search_query:
                webbrowser.open(f"{app_url}/results?search_query={search_query}")
            else:
                webbrowser.open(app_url)
        elif "://" in app_url:
            subprocess.Popen([app_url], shell=True)
        else:
            print(f"Unknown URL format: {app_url}")
    except Exception as e:
        print(f"[Execution Error] {e}")