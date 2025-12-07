import os
import sys
import subprocess
import sys
import time
import threading
import webbrowser
from winotify import Notification, audio
import urllib
import win32api
import win32gui
import win32com.shell.shell as shell
import threading
import time
import ctypes
import win32con
from win32com.shell import shellcon
import customtkinter as ctk
import psutil
import win32process




def is_window_visible_and_has_title(hwnd):
    return win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd)

# def get_open_apps_with_pids():
#     apps = []

#     def enum_window_callback(hwnd, _):
#         if is_window_visible_and_has_title(hwnd):
#             window_title = win32gui.GetWindowText(hwnd)
#             _, pid = win32process.GetWindowThreadProcessId(hwnd)
#             apps.append((window_title, pid))

#     win32gui.EnumWindows(enum_window_callback, None)
#     return apps
def get_all_running_apps():
    apps = []
    for proc in psutil.process_iter(['pid', 'name', 'exe']):
        try:
            name = proc.info['name']
            pid = proc.info['pid']
            exe = proc.info['exe']
            apps.append((name, pid, exe))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return apps

# def find_pid_by_app_name(app_name):
#     opened_apps = get_all_running_apps()
#     for name, pid, exe in opened_apps:
#         print(f"Name: {name}, PID: {pid}, EXE: {exe}")
#         if app_name.lower() in name.lower():  # case-insensitive match
#             print(f"[Local App] Found matching window '{name}' for app '{app_name}', PID: {pid}")
#             return pid
#     print(f"[Local App] No matching window found for app '{app_name}'")
#     return None

# def find_pid_by_exe_path_match(app_name):
#     """
#     Find the PID of a running process where the executable path ends with the given app_name.
#     Matches exact filename regardless of case (e.g., notepad.exe).
#     """
#     app_name = app_name.lower()
#     for proc in psutil.process_iter(['pid', 'exe']):
#         try:
#             exe_path = proc.info['exe']
#             if exe_path and os.path.basename(exe_path).lower() == app_name:
#                 print(f"[Path Match] Found PID {proc.pid} for executable '{exe_path}'")
#                 return proc.pid
#         except (psutil.NoSuchProcess, psutil.AccessDenied):
#             continue
#     print(f"[Path Match] No matching process found for '{app_name}'")
#     return None


def get_newest_parent_pid_by_app_name(app_name):
    """Return the newest PID whose name or exe includes app_name."""
    app_name = app_name.lower()
    candidates = []
    time.sleep(2)
    for proc in psutil.process_iter(['pid', 'name', 'exe', 'create_time']):
        try:
            name = (proc.info['name'] or "").lower()
            exe = (proc.info['exe'] or "").lower()
            print(f"Checking {name} ({exe})")
            if app_name in name or app_name in exe:
                print(f"[Match Found] {name} ({exe})")
                candidates.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            print(f"[Error] Access denied or process no longer exists")
            continue

    if not candidates:
        return None

    newest_proc = max(candidates, key=lambda p: p.info['create_time'])
    return newest_proc.info['pid']


try:
    # Selenium setup for web apps—allows controlled close
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    _SELENIUM_AVAILABLE = True
except ImportError:
    _SELENIUM_AVAILABLE = False

# SEARCH_PATTERNS = {
#     "youtube": "https://www.youtube.com/results?search_query={query}",
#     "spotify": "https://open.spotify.com/search/{query}",
#     "google": "https://www.google.com/search?q={query}"
# }


# Predefined WhatsApp contacts
WHATSAPP_CONTACTS = {
    "Rashmitha": "+94774335262",
    "Dinuk": "+94711570452",
    "Selani": "+94711498377",
    "Nuwani": "+94764287808",
    "Madam": "+94719693318"
}



# def show_whatsapp_popup():
#     popup = tk.Toplevel()
#     popup.title("Send WhatsApp Message")
#     popup.configure(bg="#2b2b2b")
#     popup.geometry("350x220+500+250")  # Center-ish on most screens
#     popup.attributes("-topmost", True)

#     tk.Label(popup, text="Select Contact:", bg="#2b2b2b", fg="white", font=("Segoe UI", 11)).pack(pady=(15, 5))

#     # Dropdown for names
#     selected_name = tk.StringVar(popup)
#     selected_name.set(list(WHATSAPP_CONTACTS.keys())[0])  # Default first name
#     dropdown = tk.OptionMenu(popup, selected_name, *WHATSAPP_CONTACTS.keys())
#     dropdown.config(width=20, bg="#3c3f41", fg="white", font=("Segoe UI", 10), relief="flat")
#     dropdown.pack(pady=(0, 15))

#     # Entry for message
#     tk.Label(popup, text="Enter Message:", bg="#2b2b2b", fg="white", font=("Segoe UI", 11)).pack(pady=(0, 5))
#     message_entry = tk.Entry(popup, width=30, font=("Segoe UI", 10), relief="flat", highlightthickness=1,
#                               highlightbackground="#555", highlightcolor="#888")
#     message_entry.pack(pady=(0, 15))

#     # Send button
#     def send_message():
#         name = selected_name.get()
#         phone_number = WHATSAPP_CONTACTS[name]
#         message = message_entry.get()
#         if message.strip():
#             try:
#                 pywhatkit.sendwhatmsg_instantly(phone_number, message, wait_time=15, tab_close=True, close_time=3)
#                 tk.Label(popup, text="Message Sent!", bg="#2b2b2b", fg="lime", font=("Segoe UI", 10)).pack(pady=10)
#             except Exception as e:
#                 tk.Label(popup, text=f"Error: {e}", bg="#2b2b2b", fg="red", font=("Segoe UI", 10)).pack(pady=10)

#     send_btn = tk.Button(popup, text="Send", bg="#3c3f41", fg="white", width=10, command=send_message)
#     send_btn.pack()

#     # Hover effect for button
#     def on_enter(e): send_btn.config(bg="#505354")
#     def on_leave(e): send_btn.config(bg="#3c3f41")
#     send_btn.bind("<Enter>", on_enter)
#     send_btn.bind("<Leave>", on_leave)


def show_whatsapp_popup():
    popup = ctk.CTkToplevel()
    popup.title("Send WhatsApp Message")
    popup.geometry("350x280+500+250")
    popup.attributes("-topmost", True)

    # Main Frame
    frame = ctk.CTkFrame(popup, fg_color="#1a1a1a")
    frame.pack(fill="both", expand=True, padx=20, pady=20)

    # Title Label
    ctk.CTkLabel(frame, text="Send WhatsApp Message", font=("Arial", 16, "bold")).pack(pady=(10, 15))

    # Contact Dropdown
    ctk.CTkLabel(frame, text="Select Contact:", font=("Arial", 12)).pack(anchor="w", pady=(0, 5))
    selected_name = ctk.StringVar(value=list(WHATSAPP_CONTACTS.keys())[0])
    dropdown = ctk.CTkOptionMenu(frame, variable=selected_name, values=list(WHATSAPP_CONTACTS.keys()), width=200)
    dropdown.pack(pady=(0, 15))

    # Message Entry
    ctk.CTkLabel(frame, text="Enter Message:", font=("Arial", 12)).pack(anchor="w", pady=(0, 5))
    message_entry = ctk.CTkEntry(frame, width=250, placeholder_text="Type your message here...")
    message_entry.pack(pady=(0, 15))

    # Buttons Frame
    btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
    btn_frame.pack(pady=(10, 5))

    def send_message():
        name = selected_name.get()
        message = message_entry.get().strip()

        if not message:
            messagebox.showerror("Error", "Message cannot be empty!")
            return

        phone_number = WHATSAPP_CONTACTS[name]

        try:
            pywhatkit.sendwhatmsg_instantly(phone_number, message, wait_time=1, tab_close=True, close_time=3)
            messagebox.showinfo("Success", f"Message sent to {name} successfully!")
            popup.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send message:\n{str(e)}")

    # Send Button
    send_btn = ctk.CTkButton(btn_frame, text="Send", width=100, fg_color="#4CAF50", hover_color="#388E3C",
                              command=send_message)
    send_btn.pack(side="left", padx=5)

    # Cancel Button
    cancel_btn = ctk.CTkButton(btn_frame, text="Cancel", width=100, fg_color="#F44336", hover_color="#D32F2F",
                                command=popup.destroy)
    cancel_btn.pack(side="left", padx=5)
    
def open_recommendations(chosen_recommendation: dict) -> tuple:
    from database.db import get_app_data,get_connection
    """
    Launches a local app or opens a web app with auto-close after 20 seconds
    Includes notification before closing
    """
    # get app_type from database    
    
    app_name = chosen_recommendation.get("app_name", "Unknown App")
    app_url = chosen_recommendation.get("app_url", "")
    search_query = chosen_recommendation.get("search_query", "")
    is_local = chosen_recommendation.get("is_local", False)
    conn = get_connection()

    print("Chosen Recommendation:", chosen_recommendation)
    app_info = get_app_data(conn, app_name.lower())
    print("[App Data]:", app_info)

    def send_reminder_notification():
        """Send reminder notification before closing"""
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "..", "assets", "res", "Icon.ico")
            icon_path = os.path.abspath(icon_path) if os.path.exists(icon_path) else None
            
            toast = Notification(
                app_id="EMOFI",
                title="Time's Up!",
                msg=f"Closing {app_name} to help you focus",
                duration="long",
                icon=icon_path
            )
            toast.add_actions(label="Got it")
            toast.set_audio(audio.Mail, loop=True)
            toast.show()
            print("[Notification] Sent reminder")
        except Exception as e:
            print(f"[Notification Error] {e}")

    # def close_local_app(pid):
    #     """Helper to close local app and its children"""
    #     try:
    #         # Send reminder notification
    #         send_reminder_notification()
            
    #         # Give user a moment to see notification
    #         time.sleep(2)
            
    #     #     # Terminate process
    #     #     proc = psutil.Process(pid)
    #     #     for child in proc.children(recursive=True):
    #     #         try:
    #     #             child.kill()
    #     #         except psutil.NoSuchProcess:
    #     #             pass
    #     #     proc.kill()
    #     #     print(f"[Auto-Close] Closed local app (PID: {pid})")
    #     # except Exception as e:
    #     #     print(f"[Auto-Close] Error closing app: {e}")
        
    #         # Using psutil to terminate the process more gracefully
    #         proc = psutil.Process(pid)
    #         proc.terminate()  # sends SIGTERM equivalent
    #         proc.wait(timeout=5)  # wait for it to exit
    #         print(f"Process {pid} terminated.")
    #     except psutil.NoSuchProcess:
    #         print(f"No process found with PID {pid}.")
    #     except psutil.TimeoutExpired:
    #         print(f"Process {pid} did not terminate in time, killing it.")
    #         proc = psutil.Process(pid)
    #         proc.kill()  # force kill
    #     except Exception as e:
    #         print(f"Error terminating process {pid}: {e}")
            
    def close_app_by_pid(pid):
        try:
            proc = psutil.Process(pid)

            # Get all child processes (recursively)
            children = proc.children(recursive=True)

            # Terminate children first
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass

            # Wait for children to terminate
            psutil.wait_procs(children, timeout=5)

            # Now terminate the main process
            proc.terminate()
            proc.wait(timeout=5)
            print(f"Process {pid} and its children terminated.")

        except psutil.NoSuchProcess:
            print(f"No process found with PID {pid}.")

        except psutil.TimeoutExpired:
            print(f"Process {pid} did not terminate in time, killing it and its children.")
            # Kill remaining children
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            proc.kill()

        except Exception as e:
            print(f"Error terminating process {pid}: {e}")

    def close_web_driver(driver):
        """Helper to close web driver"""
        try:
            # Send reminder notification
            send_reminder_notification()
            
            # Give user a moment to see notification
            time.sleep(2)
            
            # Close browser
            driver.quit()
            print("[Auto-Close] Closed web browser")
        except Exception as e:
            print(f"[Auto-Close] Error closing browser: {e}")

    def build_url(app_url: str, search_query: str) -> str:
        """Build full URL with search query"""
        if "<search_query>" in app_url:
            if search_query:
                encoded_query = urllib.parse.quote(search_query.strip())
                return app_url.replace("<search_query>", encoded_query)
            return app_url.replace("<search_query>", "")
        elif search_query:
            delimiter = "&" if "?" in app_url else "?"
            return f"{app_url}{delimiter}search_query={urllib.parse.quote(search_query.strip())}"
        return app_url
    
    def unified_app_launcher(app_info):
        try:
            if app_info["app_type"] == "uwp":
                app_id = app_info["app_id"]
                subprocess.Popen(f'explorer shell:AppsFolder\\{app_id}', shell=True)
                print(f"[Launch] UWP app: {app_info['name']}")
                return True

            elif app_info["app_type"] == "classic":
                exe_path = app_info["path"]
                if not exe_path or not os.path.isfile(exe_path):
                    print(f"[Error] Executable not found: {exe_path}")
                    return False
                
                shell.ShellExecuteEx(
                        lpVerb='runas',
                        lpFile=exe_path,
                        lpParameters='',
                        nShow=win32con.SW_SHOWNORMAL
                )
                print(f"[Launch] Classic app as Admin: {exe_path}")
                return True

        except Exception as e:
            print(f"[Launch Error] {e}")
            return False


    # 1) Local app path
    if is_local:
        try:
            unified_app_launcher(app_info)
            print(f"[Admin Launch] Launched")
            print(f"[Local App] Launched {app_name}")
            found_pid = get_newest_parent_pid_by_app_name(app_name)
            print(f"[Local App] Found PID: {found_pid} for app '{app_name}'")
            if not found_pid:
                print(f"[Local App] No matching PID found for {app_name}")
                return False, None, None

            # Start auto-close timer
            threading.Timer(20.0, close_app_by_pid, args=(found_pid,)).start()

            return True, found_pid, 'local'
        except Exception as ex:
            print(f"Error launching local app: {ex}")
            return False, None, None

    # 2) Web app
    else:
        if not app_url.startswith(("http://", "https://")):
            print(f"Error: Invalid URL for web app '{app_name}': {app_url}")
            return False, None, None

        url = build_url(app_url, search_query)

        # check if app name is WhatsApp
        if app_name.lower() == "whatsapp":
            show_whatsapp_popup()
            # wait for 15 seconds to allow user to send message
            time.sleep(15)

        # Use Selenium for web apps to enable auto-close
        if _SELENIUM_AVAILABLE:
            try:
                options = Options()
                options.add_argument("--start-maximized")
                driver = webdriver.Chrome(options=options)
                driver.get(url)
                print(f"[Selenium] Opened {app_name} at {url}")
                
                # Start auto-close timer
                threading.Timer(20.0, close_web_driver, args=(driver,)).start()
                
                return True, driver, 'web'
            except Exception as ex:
                print(f"Selenium error: {ex}")
                # Fall through to webbrowser method

        # Fallback to default browser (no auto-close)
        try:
            webbrowser.open(url)
            print(f"[Webbrowser] Opened {app_name} at {url}")
            
            # Send notification that we can't auto-close
            try:
                toast = Notification(
                    app_id="EMOFI",
                    title="No Auto-Close",
                    msg=f"Opened in default browser. Close manually when done.",
                    duration="long"
                )
                toast.set_audio(audio.Mail, loop=True)
                toast.show()
            except Exception:
                pass
            
            return True, None, 'web'
        except Exception as ex:
            print(f"Webbrowser error: {ex}")
            return False, None, None
            
    return False, None, None

def resource_path(rel_path):
    """ Get absolute path to resource, works for dev and for PyInstaller bundle """
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base_path, rel_path)


