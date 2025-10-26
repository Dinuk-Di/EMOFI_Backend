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

def send_notification(recommendation,timeout=200):
    user_action = False
    print(f"[notification] {recommendation}")
    

    if not os.path.exists(icon_path):
        print(f"Warning: Icon file not found at {icon_path}")
        icon = None
    else:
        icon = {
        'src': icon_path,
        'placement': 'appLogoOverride'
        }

    def on_click(*args):
        try:
            nonlocal user_action
            user_action = True
        except Exception as e:
            print(f"Error executing script: {e}")

    # def on_click():
    #     try:
    #         print("Clicked")
    #     except Exception as e:
    #         print(f"Error executing script: {e}")

    try:
        toast('Emotion Recognition Test ', str(recommendation) , icon=icon,  app_id="EMOFI", on_click=on_click, button='Dismiss')
        print("notification sent")
    except Exception as e:
        print(f"Error sending notification: {e}")
    return user_action


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