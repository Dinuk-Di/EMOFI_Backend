from typing import List
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import win32api
import win32con
import time
import subprocess
import webbrowser
from win11toast import toast
import threading
icon_path = r'D:\RuhunaNew\Academic\Research\Facial_Recog_Repo\Group_50_Repo\DesktopApp\assets\res\Icon.jpg'
executer_path = r'executer.pyw'
import tkinter as tk
import os
import winsound


import tkinter as tk

# def send_notification(title, recommended_output, recommended_options, timeout=None):
#     winsound.MessageBeep()
#     selected_app = {"value": None}

#     def show_app_options(app_options):
#         for widget in frame.winfo_children():
#             widget.destroy()

#         tk.Label(frame, text="Choose an app:", bg="#2b2b2b", fg="white", font=("Segoe UI", 10)).pack(pady=(0, 10))

#         for option in app_options:
#             btn = tk.Button(frame, text=option.app_name, width=25, bg="#3c3f41", fg="white", font=("Segoe UI", 9),
#                             relief="flat", highlightthickness=0, command=lambda opt=option: handle_app_selection(opt))
#             btn.pack(pady=3)

#     def handle_app_selection(option):
#         # If local app, select immediately
#         if option.is_local:
#             select_app(option, None)
#         else:
#             # Show input field for search query
#             show_search_input(option)

#     def show_search_input(option):
#         for widget in frame.winfo_children():
#             widget.destroy()

#         tk.Label(frame, text=f"Enter search text for {option.app_name} (or skip):", bg="#2b2b2b", fg="white",
#                  font=("Segoe UI", 10), wraplength=280).pack(pady=(10, 5))

#         query_entry = tk.Entry(frame, width=25, font=("Segoe UI", 10))
#         query_entry.pack(pady=10)

#         btn_frame = tk.Frame(frame, bg="#2b2b2b")
#         btn_frame.pack(pady=10)

#         # Confirm with query
#         tk.Button(btn_frame, text="Search", bg="#3c3f41", fg="white", width=10,
#                   command=lambda: select_app(option, query_entry.get())).pack(side="left", padx=5)

#         # Skip query
#         tk.Button(btn_frame, text="Skip", bg="#3c3f41", fg="white", width=10,
#                   command=lambda: select_app(option, None)).pack(side="left", padx=5)

#     def select_app(option, custom_query):
#         selected_app["value"] = {
#             "app_name": option.app_name,
#             "app_url": option.app_url,
#             "search_query": custom_query if custom_query else option.search_query,
#             "is_local": option.is_local,
#             "category": option.category
#         }
#         root.destroy()

#     def select_recommendation(index):
#         show_app_options(recommended_options[index])

#     # Create main window
#     root = tk.Tk()
#     root.title(title)
#     root.overrideredirect(True)
#     root.attributes("-topmost", True)
#     root.configure(bg="#2b2b2b")

#     width, height = 320, 250
#     x = root.winfo_screenwidth() - width - 20
#     y = 50
#     root.geometry(f"{width}x{height}+{x}+{y}")

#     frame = tk.Frame(root, bg="#2b2b2b", bd=2)
#     frame.place(relwidth=1, relheight=1)

#     tk.Label(frame, text=title, bg="#2b2b2b", fg="white", font=("Segoe UI", 12, "bold")).pack(pady=(10, 5))
#     tk.Label(frame, text="Choose an option:", bg="#2b2b2b", fg="white", font=("Segoe UI", 10)).pack(pady=(0, 10))

#     for idx, rec in enumerate(recommended_output):
#         tk.Button(frame, text=rec, width=25, bg="#3c3f41", fg="white", font=("Segoe UI", 9),
#                   relief="flat", highlightthickness=0, command=lambda i=idx: select_recommendation(i)).pack(pady=3)

#     if timeout:
#         root.after(timeout, root.destroy)

#     root.mainloop()
#     return selected_app["value"]



def send_notification(title, recommended_output, recommended_options, timeout=None):
    winsound.MessageBeep()
    selected_app = {"value": None}

    # --- Configuration ---
    bg_color = "#2b2b2b"       # Main background
    btn_normal = "#3c3f41"     # Button default color
    btn_hover = "#4a4d50"      # Button hover color
    accent_color = "#0e3aa9"   # Color for primary actions (like Search)
    text_color = "white"
    
    # --- Helper: Button Hover Effects ---
    def on_enter(e):
        e.widget['bg'] = btn_hover
    def on_leave(e):
        e.widget['bg'] = btn_normal

    def close_window():
        root.destroy()

    # --- View 3: Handle Final Selection ---
    def select_app(option, custom_query):
        selected_app["value"] = {
            "app_name": option.app_name,
            "app_url": option.app_url,
            "search_query": custom_query if custom_query else option.search_query,
            "is_local": option.is_local,
            "category": option.category
        }
        root.destroy()

    # --- View 2b: Search Input (for web apps) ---
    def show_search_input(option):
        for widget in content_frame.winfo_children():
            widget.destroy()

        # Title with padding
        tk.Label(content_frame, text=f"Search in {option.app_name}", 
                 bg=bg_color, fg=text_color, font=("Segoe UI", 11, "bold")).pack(pady=(0, 10))

        tk.Label(content_frame, text="Enter search topic (optional):", 
                 bg=bg_color, fg="#cccccc", font=("Segoe UI", 9)).pack(pady=(0, 5))

        # Stylized Entry
        query_entry = tk.Entry(content_frame, width=30, font=("Segoe UI", 10), 
                               bg="#1e1e1e", fg="white", insertbackground="white", relief="flat")
        query_entry.pack(pady=10, ipady=5) # ipady makes the box taller

        btn_frame = tk.Frame(content_frame, bg=bg_color)
        btn_frame.pack(pady=15)

        # Search Button
        search_btn = tk.Button(btn_frame, text="Search", bg=accent_color, fg="white", width=10,
                               font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2",
                               command=lambda: select_app(option, query_entry.get()))
        search_btn.pack(side="left", padx=5)

        # Skip Button
        skip_btn = tk.Button(btn_frame, text="Skip", bg=btn_normal, fg="white", width=10,
                             font=("Segoe UI", 9), relief="flat", cursor="hand2",
                             command=lambda: select_app(option, None))
        skip_btn.pack(side="left", padx=5)
        
        # Add hover to skip button
        skip_btn.bind("<Enter>", on_enter)
        skip_btn.bind("<Leave>", on_leave)

    def handle_app_selection(option):
        if option.is_local:
            select_app(option, None)
        else:
            show_search_input(option)

    # --- View 2a: App List ---
    def show_app_options(app_options):
        for widget in content_frame.winfo_children():
            widget.destroy()

        tk.Label(content_frame, text="Choose an app to open:", 
                 bg=bg_color, fg=text_color, font=("Segoe UI", 11)).pack(pady=(0, 15))

        for option in app_options:
            btn = tk.Button(content_frame, text=option.app_name, width=30, 
                            bg=btn_normal, fg=text_color, font=("Segoe UI", 10),
                            relief="flat", cursor="hand2",
                            command=lambda opt=option: handle_app_selection(opt))
            
            # Add Internal Padding (ipady) to make buttons taller
            btn.pack(pady=5, ipady=4)
            
            # Bind hover events
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)

    # --- View 1: Main Recommendations ---
    def select_recommendation(index):
        show_app_options(recommended_options[index])

    def show_main_menu():
        tk.Label(content_frame, text=title, bg=bg_color, fg=text_color, 
                 font=("Segoe UI", 12, "bold"), wraplength=280).pack(pady=(0, 15))
        
        tk.Label(content_frame, text="Select a suggestion:", 
                 bg=bg_color, fg="#cccccc", font=("Segoe UI", 9)).pack(pady=(0, 10))

        for idx, rec in enumerate(recommended_output):
            btn = tk.Button(content_frame, text=rec, width=30, 
                            bg=btn_normal, fg=text_color, font=("Segoe UI", 10),
                            relief="flat", cursor="hand2",
                            command=lambda i=idx: select_recommendation(i))
            
            # Add Internal Padding (ipady)
            btn.pack(pady=4, ipady=4)
            
            # Bind hover events
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)

    # --- Main Window Setup ---
    root = tk.Tk()
    root.title(title)
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg=bg_color)

    # --- Positioning (Bottom Right) ---
    width, height = 350, 320  # Increased size slightly for padding
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Calculate X and Y
    # 20px margin from Right
    x = screen_width - width - 20
    # 70px margin from Bottom (Accounts for Windows Taskbar)
    y = screen_height - height - 70 
    
    root.geometry(f"{width}x{height}+{x}+{y}")

    # --- Top Bar (Close Button) ---
    top_frame = tk.Frame(root, bg=bg_color, height=30)
    top_frame.pack(fill="x", side="top")
    
    close_btn = tk.Button(top_frame, text="✕", bg=bg_color, fg="#888888", 
                          bd=0, font=("Arial", 11), command=close_window, 
                          activebackground="#cf2323", activeforeground="white")
    close_btn.pack(side="right", padx=10, pady=5)
    
    # Hover effect for close button
    close_btn.bind("<Enter>", lambda e: e.widget.config(bg="#cf2323", fg="white"))
    close_btn.bind("<Leave>", lambda e: e.widget.config(bg=bg_color, fg="#888888"))

    # --- Content Frame (With Padding) ---
    # This frame holds all the changing content (Menus, Inputs, etc.)
    content_frame = tk.Frame(root, bg=bg_color)
    content_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    # Initialize
    show_main_menu()

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