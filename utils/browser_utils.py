import os
import subprocess
import webbrowser
import win32api
import win32con
import psutil
import time
import winapps
from typing import Optional, Dict, List
import threading
import uuid
import json
import sqlite3
from datetime import datetime, timedelta
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from old_utils.notifications import send_notification
from old_utils.contactWindowInterface import ContactWindowInterface
from DesktopApp.old_utils.app_config import kNOWN_APPS_LIST
from DesktopApp.old_utils.communication_apps_config import COMMUNICATION_APPS_LIST

launched_apps = {}
opened_browser_instances = []  # Track browser instances we launched
opened_browser_tabs: List[Dict] = []  # Track individual tabs
# Store opened apps info with timestamps
opened_apps_info = []  # Each item: dict with keys: type='desktop'|'web', process=psutil.Process, opened_at=timestamp, driver=webdriver (optional)

# Timeout duration in seconds (15 minutes)
APP_TIMEOUT_SECONDS = 30
APP_WARNING_SECONDS = 20

def get_browser_path(browser_name: str) -> Optional[str]:
    """Get the path to the browser executable based on name"""
    browser_name = browser_name.lower()
    paths = {
        'chrome': r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        'msedge': r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        'firefox': r"C:\Program Files\Mozilla Firefox\firefox.exe",
        'opera': r"C:\Program Files\Opera\launcher.exe"
    }
    
    for name, path in paths.items():
        if browser_name in name:
            if os.path.exists(path):
                return path
    return None

def is_app_installed(app_name: str) -> bool:
    app_name = app_name.lower()
    if app_name in COMMUNICATION_APPS_LIST:
        process_name = COMMUNICATION_APPS_LIST[app_name]["process"]
        if any(proc.info['name'].lower() == process_name.lower() 
               for proc in psutil.process_iter(['name'])):
            return True
        
        try:
            return any(app_name in app.name.lower() 
                       for app in winapps.list_installed())
        except ImportError:
            print("winapps not available, limited installation detection")
    
    return False

def close_tab_after_delay(driver, delay=5):
    """Close the browser tab after a delay without closing the entire browser"""
    def close_tab():
        time.sleep(delay)
        try:
            # Close the current tab only
            driver.close()
            print(f"Closed tab after {delay} seconds")
            
            # Remove from tracking lists
            global opened_apps_info, opened_browser_tabs
            for app_info in opened_apps_info[:]:
                if app_info.get('driver') == driver:
                    opened_apps_info.remove(app_info)
            for tab_info in opened_browser_tabs[:]:
                if tab_info.get('driver') == driver:
                    opened_browser_tabs.remove(tab_info)
        except Exception as e:
            print(f"Error closing tab: {str(e)}")
    
    threading.Thread(target=close_tab, daemon=True).start()

def open_recommendation(recommendation: dict) -> str:
    global launched_apps, opened_browser_tabs, opened_apps_info

    print(f"[Open_recommendation] {recommendation}")
    url = recommendation.get("app_url", "")
    app_name = recommendation.get("app_name", "")
    app_name_lower = app_name.lower()

    for app_info in kNOWN_APPS_LIST:
        # Check for a match (case-insensitive, allows partial for longer names)
        if app_name_lower == app_info["name"].lower() or \
           (app_name_lower in app_info["name"].lower() and len(app_name_lower) > 2):

            # Desktop app launch path
            process = None
            if "aumid" in app_info and app_info["aumid"]:
                aumid_to_use = app_info["aumid"]
                try:
                    win32api.ShellExecute(
                        0,                      # hwnd: handle to parent window (0 for no parent)
                        "open",                 # operation: "open", "print", "edit", "explore", "find"
                        "explorer.exe",         # file: The program to execute (explorer.exe for shell:Appsfolder)
                        aumid_to_use,           # parameters: The AUMID as a shell URI
                        None,                   # directory: default directory
                        win32con.SW_SHOWNORMAL  # show command: how the application is shown
                    )

                    time.sleep(2)  # Give app time to launch
                    processes = []
                    for proc in psutil.process_iter(['name']):
                        if proc.info['name'].lower() == app_info["process"].lower():
                            processes.append(proc)

                    if processes:
                        opened_apps_info.append({
                            'type': 'desktop',
                            'processes': processes,
                            'app_name': app_info["name"],
                            'opened_at': time.time()
                        })
                        print("opened app info aumid process: ", opened_apps_info)
                    start_monitoring_thread()
                    return f"Successfully launched {app_info['name']} via AUMID: '{aumid_to_use}'."
                except Exception as e:
                    return f"Error launching {app_info['name']} via AUMID '{aumid_to_use}': {e}"
            elif "location" in app_info and app_info["location"]:    
                app_path_to_use = app_info["location"]

                if not app_path_to_use:
                    return f"Error: No path provided for {app_info['name']} in the list."

                # Attempt to launch the app using the provided path
                try:
                    win32api.ShellExecute(
                        0,                           # hwnd
                        "open",                      # operation
                        app_path_to_use,             # file: the path to the executable or shortcut
                        None,                        # parameters (not needed for simple open)
                        None,                        # directory
                        win32con.SW_SHOWNORMAL       # show command
                    )
                    
                    time.sleep(2)  # Give app time to launch
                    matching_procs = []
                    for proc in psutil.process_iter(['name']):
                        try:
                            if proc.info['name'].lower() == app_info["process"].lower():
                                matching_procs.append(proc)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue

                    if matching_procs:
                        # Save the list of processes instead of single process
                        opened_apps_info.append({
                            'type': 'desktop',
                            'processes': matching_procs,
                            'app_name': app_info["name"],
                            'opened_at': time.time()
                        })
                        print("opened_app_info : ",opened_apps_info)
                    start_monitoring_thread()
                    return f"Successfully launched {app_info['name']} using path: {app_path_to_use}."
                except Exception as e:
                    return f"Error launching {app_info['name']} from {app_path_to_use}: {e}"
            else:
                return f"Error: No valid launch method (location or aumid) provided for '{app_info['name']}' in the 'known_apps_list'."
    
    # Handle web URLs
    if url.startswith(("http://", "https://")):
        try:
            # Add search query if present
            if recommendation.get("search_query"):
                search = recommendation["search_query"].replace(" ", "+")
                url += f"/results?search_query={search}"
                
            # Open new Selenium browser window each time
            print("Web URL: ", url)
            options = webdriver.ChromeOptions()
            options.add_argument("--new-window")  # Open in new window
            driver = webdriver.Chrome(options=options)
            driver.get(url)

            # Store tab information
            tab_info = {
                'url': url,
                'driver': driver,
                'opened_at': time.time()
            }
            
            # Add to tracking lists
            opened_browser_tabs.append(tab_info)
            opened_apps_info.append({
                'type': 'web',
                'driver': driver,
                'url': url,
                'opened_at': time.time()
            })

            # Close this specific tab after 5 seconds
            close_tab_after_delay(driver, 5)
            
            start_monitoring_thread()
            return f"Opened {url} in new browser tab. Will close in 5 seconds."
        except Exception as e:
            # Fallback to webbrowser module if Selenium fails
            webbrowser.open(url)
            return f"Failed to open URL '{url}' with Selenium. Used fallback: {e}"

    return f"Recommendation '{recommendation}' is neither a recognized URL nor a known application."

def terminate_process_tree(proc):
    try:
        children = proc.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except Exception:
                pass
        proc.terminate()
        gone, alive = psutil.wait_procs([proc] + children, timeout=5)
        for p in alive:
            try:
                p.kill()
            except Exception:
                pass
    except Exception as e:
        print(f"Error terminating process tree: {e}")

def close_tracked_app(app_info):
    try:
        if app_info['type'] == 'desktop':
            for proc in app_info.get('processes', []):
                terminate_process_tree(proc)
            print(f"Closed desktop app: {app_info.get('app_name')}")
            return True
            
        elif app_info['type'] == 'web':
            # For web apps, we don't need to close the entire browser here
            # because the tab-closing thread will handle individual tabs
            print(f"Web app already scheduled for closing: {app_info.get('url')}")
            return True
    except Exception as e:
        print(f"Error closing app: {e}")
    return False

def monitor_opened_apps():
    current_time = time.time()
    to_remove = []
    is_closed = False
    closing_text = "Time to get back to work"
    
    for app_info in opened_apps_info:
        elapsed = current_time - app_info['opened_at']
        
        # Check for warning time (20 seconds)
        if abs(elapsed - APP_WARNING_SECONDS) <= 1: 
            user_action = send_notification(closing_text)
            if user_action:
                is_closed = close_tracked_app(app_info)
                if is_closed:
                    to_remove.append(app_info)
        
        # Check for timeout (30 seconds)
        elif elapsed >= APP_TIMEOUT_SECONDS:
            is_closed = close_tracked_app(app_info)
            if is_closed:
                to_remove.append(app_info)
        
    # Remove closed apps from tracking
    for app_info in to_remove:
        if app_info in opened_apps_info:
            opened_apps_info.remove(app_info)
            
    return is_closed

def start_monitoring_thread(interval_sec=60):
    def monitor_loop():
        while True:
            try:
                monitor_opened_apps()
                time.sleep(interval_sec)
            except Exception as e:
                print(f"Monitoring error: {str(e)}")
    
    print("Starting monitoring thread")
    thread = threading.Thread(target=monitor_loop, daemon=True)
    thread.start()
    print("Monitoring active")

def find_process_by_name(process_name: str, timeout: int = 5) -> Optional[psutil.Process]:
    """Helper to find a process by name with timeout"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        for proc in psutil.process_iter(['name']):
            if proc.info['name'].lower() == process_name.lower():
                return proc
        time.sleep(0.5)
    return None