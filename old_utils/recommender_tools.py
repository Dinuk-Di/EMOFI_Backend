import os
import subprocess
import webbrowser
import win32api # New import
import win32con # New import
import psutil
import time
import winapps
from dotenv import load_dotenv
from typing import Optional,Dict, List
import webbrowser
from old_utils.contactWindowInterface import ContactWindowInterface
import uuid
from old_utils.app_config import kNOWN_APPS_LIST
from old_utils.communication_apps_config import COMMUNICATION_APPS_LIST
import json
import sqlite3
from datetime import datetime, timedelta
import os
from pathlib import Path
import re
from telethon.sync import TelegramClient
from telethon.tl import functions  # Add this with other imports
from telethon.tl.functions.contacts import GetContactsRequest
from telethon.tl.functions.messages import GetHistoryRequest
import asyncio
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from old_utils.notifications import send_notification

import time
import threading

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


            opened_browser_tabs.append({
                'url': url,
                'browser': 'chrome',
                'method': 'selenium',
                'driver': driver,
                'window_handle': driver.current_window_handle,
                'opened_at': time.time()
            })

            opened_apps_info.append({
                'type': 'web',
                'driver': driver,
                'url': url,
                'opened_at': time.time()
            })
            print("opened app info: ", opened_apps_info)
            start_monitoring_thread()
            return f"Opened {url} in default browser tab."
        except Exception as selenium_error:
            webbrowser.open(url)
            return f"Failed to open URL '{url}': {e}"

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
        elif app_info['type'] == 'web':
            driver = app_info.get('driver')
            if driver:
                driver.quit()
                return True 
            print(f"Closed browser window for URL: {app_info.get('url')}")
    except Exception as e:
        print(f"Error closing app: {e}")

def monitor_opened_apps():
    current_time = time.time()
    to_remove = []
    is_closed = False
    closing_text = "Time to get back to work"
    
    for app_info in opened_apps_info:
        if abs((current_time - app_info['opened_at']) - APP_WARNING_SECONDS) <= 1: 
            user_action = send_notification(closing_text)
            if user_action:
                is_closed = close_tracked_app(app_info)
                to_remove.append(app_info)
        elif current_time - app_info['opened_at'] >= APP_TIMEOUT_SECONDS:
            is_closed = close_tracked_app(app_info)
            to_remove.append(app_info)
        
    for app_info in to_remove:
        opened_apps_info.remove(app_info)
    return is_closed

def start_monitoring_thread(interval_sec=60):
    def monitor_loop():
        app_is_closed = False
        while not app_is_closed:
            app_is_closed = monitor_opened_apps()
    print("start monitoring")
    thread = threading.Thread(target=monitor_loop, daemon=False)
    thread.start()
    print("opened app info: ", opened_apps_info)

def find_process_by_name(process_name: str, timeout: int = 5) -> Optional[psutil.Process]:
    """Helper to find a process by name with timeout"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        for proc in psutil.process_iter(['name']):
            if proc.info['name'].lower() == process_name.lower():
                return proc
        time.sleep(0.5)
    return None


def get_whatsapp_contacts():
    """Extracts recent WhatsApp contacts from local database"""
    contacts = []
    whatsapp_db_path = os.path.join(os.getenv('LOCALAPPDATA'), 'WhatsApp', 'Databases', 'msgstore.db')
    
    if not os.path.exists(whatsapp_db_path):
        return contacts

    try:
        conn = sqlite3.connect(f"file:{whatsapp_db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        
        # Query last 3 chats
        cursor.execute("""
            SELECT chat.subject, message.timestamp, jid.raw_string 
            FROM message
            JOIN chat ON message.chat_row_id = chat._id
            JOIN jid ON chat.jid_row_id = jid._id
            ORDER BY message.timestamp DESC
            LIMIT 3
        """)
        
        for subject, timestamp, raw_jid in cursor.fetchall():
            phone = raw_jid.split('@')[0]
            contacts.append({
                "name": subject or f"Contact {phone}",
                "phone": phone,
                "id": raw_jid,
                "last_contact": format_timestamp(timestamp)
            })
            
    except Exception as e:
        print(f"Error reading WhatsApp DB: {e}")
    finally:
        conn.close()
    
    return contacts

def get_telegram_contacts():
    """Extracts recent Telegram contacts"""
    contacts = []
    api_id = os.getenv('API_ID')
    api_hash = os.getenv('API_HASH')
    phone = os.getenv('API_PHONE')
    telegram_path = os.path.join(os.getenv('APPDATA'), 'Telegram Desktop', 'tdata')
    
    async def get_recent_chats():
        async with TelegramClient('session_name', api_id, api_hash) as client:
            await client.start(phone)
            
            # Get recent dialogs (chats)
            dialogs = await client.get_dialogs(limit=10)
            
            for dialog in dialogs:
                if dialog.is_user:  # Only individual chats
                    entity = dialog.entity
                    contacts.append({
                        "name": entity.first_name + (f" {entity.last_name}" if entity.last_name else ""),
                        "id": entity.username or str(entity.id),
                        "phone": getattr(entity, 'phone', ''),
                        "last_contact": format_timestamp(dialog.date.timestamp()),
                        "has_called": await check_recent_calls(client, entity.id)
                    })
            
            return contacts
    async def check_recent_calls(client, user_id):
        """Check if recent calls exist with this user"""
        try:
            result = await client(functions.phone.GetRecentCallersRequest())
            return any(caller.peer.user_id == user_id for caller in result.callers)
        except:
            return False
    
    try:
        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        contacts = loop.run_until_complete(get_recent_chats())
        print("contact loop is runing")
    except Exception as e:
        print(f"Error getting Telegram contacts: {e}")
    print("Finished contact loop: ", contacts)
    return contacts

def get_teams_contacts():
    """Extracts recent Microsoft Teams contacts"""
    contacts = []
    teams_db_path = os.path.join(os.getenv('APPDATA'), 'Microsoft', 'Teams', 'IndexedDB')
    
    if not os.path.exists(teams_db_path):
        return contacts

    try:
        # Teams stores data in IndexedDB - this is a simplified approach
        for db_file in Path(teams_db_path).glob('*.teams.microsoft.com*.ldb'):
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%conversations%'")
            if cursor.fetchone():
                cursor.execute("""
                    SELECT displayName, lastMessageTimestamp 
                    FROM conversations 
                    ORDER BY lastMessageTimestamp DESC 
                    LIMIT 3
                """)
                
                for name, timestamp in cursor.fetchall():
                    contacts.append({
                        "name": name,
                        "id": name.lower().replace(' ', '') + "@teams",
                        "last_contact": format_timestamp(timestamp)
                    })
                    
            conn.close()
            
    except Exception as e:
        print(f"Error reading Teams DB: {e}")
    
    return contacts

def format_timestamp(timestamp):
    """Convert various timestamp formats to human-readable"""
    try:
        if timestamp > 1e12:  # WhatsApp timestamp (microseconds)
            dt = datetime.fromtimestamp(timestamp/1000)
        elif timestamp > 1e9:  # Unix timestamp
            dt = datetime.fromtimestamp(timestamp)
        else:  # Teams timestamp (milliseconds)
            dt = datetime.fromtimestamp(timestamp/1000)
            
        if dt.date() == datetime.today().date():
            return dt.strftime("%H:%M")
        elif dt.date() == (datetime.today() - timedelta(days=1)).date():
            return "Yesterday"
        else:
            return dt.strftime("%Y-%m-%d")
    except:
        return "Recently"
    

def get_recent_contacts(app_name: str) -> list:
    """Get real recent contacts for the specified app"""
    app_name = app_name.lower()
    
    if 'whatsapp' in app_name:
        return get_whatsapp_contacts()
    elif 'telegram' in app_name:
        print("Came to get recent contacts in telegram")
        return get_telegram_contacts()
    elif 'teams' in app_name:
        return get_teams_contacts()
    
    # Fallback mock data
    return [
        {"name": "John Doe", "phone": "+1234567890", "id": "john123", "last_contact": "2h ago"},
        {"name": "Jane Smith", "phone": "+1987654321", "id": "jane456", "last_contact": "Yesterday"}
    ]