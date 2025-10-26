import webbrowser
import psutil
import win32gui
import win32con
import win32process
import time
import threading
import subprocess
import urllib.request
import json
import re
from selenium import webdriver
from win11toast import toast

# Browser configurations
BROWSER_CONFIG = {
    'chrome': {
        'exe': 'chrome.exe',
        'path': r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        'debug_port': 9222
    },
    'firefox': {
        'exe': 'firefox.exe',
        'path': r"C:\Program Files\Mozilla Firefox\firefox.exe"
    },
    'msedge': {
        'exe': 'msedge.exe',
        'path': r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    },
    'opera': {
        'exe': 'opera.exe',
        'path': r"C:\Program Files\Opera\launcher.exe"
    },
    'brave': {
        'exe': 'brave.exe',
        'path': r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
    }
}

# Global variables
browser_pid = None
driver = None
browser_name = None

def is_browser_running(browser_name):
    """Check if a specific browser is running"""
    exe_name = BROWSER_CONFIG[browser_name]['exe']
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] and exe_name.lower() == proc.info['name'].lower():
            return proc.info['pid']
    return None

def any_browser_running():
    """Check if any supported browser is running"""
    for browser in BROWSER_CONFIG:
        pid = is_browser_running(browser)
        if pid:
            return browser, pid
    return None, None

def bring_browser_to_front(pid):
    """Bring browser window to front by PID"""
    def enumHandler(hwnd, lParam):
        if win32gui.IsWindowVisible(hwnd):
            _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
            if found_pid == pid:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                return False
        return True
    win32gui.EnumWindows(enumHandler, None)

def open_url_in_browser(url, browser_name):
    """Open URL in a specific browser"""
    browser_path = BROWSER_CONFIG[browser_name]['path']
    webbrowser.register(browser_name, None, webbrowser.BackgroundBrowser(browser_path))
    webbrowser.get(browser_name).open_new_tab(url)

def get_all_tabs(port=9222):
    """Get all open tabs using Chrome's debugging protocol"""
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/json", timeout=2) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error getting tabs: {e}")
        return []

def find_youtube_tab(tabs):
    """Find YouTube tab by URL pattern"""
    for tab in tabs:
        if 'url' in tab and re.search(r"youtube\.com|youtu\.be", tab['url']):
            return tab
    return None

def ensure_chrome_debugging():
    """Ensure Chrome is running with debugging enabled"""
    try:
        # Check if debugging is available
        urllib.request.urlopen("http://localhost:9222/json", timeout=1)
        return True
    except:
        # Start Chrome with debugging
        chrome_path = BROWSER_CONFIG['chrome']['path']
        subprocess.Popen([
            chrome_path,
            "--remote-debugging-port=9222",
            "--new-window",
            "about:blank"
        ])
        time.sleep(3)  # Give Chrome time to start
        return True

def show_emofi_toast():
    """Show EMOFI toast after 10 seconds"""
    time.sleep(10)
    toast("EMOFI", "Time to get back to work!", duration='long')

def open_with_selenium(url):
    """Open URL with Selenium"""
    options = webdriver.ChromeOptions()
    options.add_argument("--new-window")
    return webdriver.Chrome(options=options)

def main():
    global browser_pid, driver, browser_name
    
    url = "https://youtube.com"
    
    # Check if any browser is running
    result = any_browser_running()
    if result[0]:
        browser_name, browser_pid = result
        print(f"Using existing browser: {browser_name}")
        open_url_in_browser(url, browser_name)
        
        # Bring browser to front
        bring_browser_to_front(browser_pid)
    else:
        print("No browsers running. Using Selenium Chrome")
        driver = open_with_selenium(url)
        driver.get(url)
        browser_name = 'chrome'
    
    # Start thread to show toast after 10 seconds
    threading.Thread(target=show_emofi_toast, daemon=True).start()
    
    # Keep script running temporarily to allow toast to appear
    time.sleep(15)

if __name__ == "__main__":
    # Install required package if needed
    try:
        import win11toast
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "win11toast"])
        from win11toast import toast
        
    main()