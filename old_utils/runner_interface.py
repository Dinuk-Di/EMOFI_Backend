import ctypes
import sys
from old_utils.mainWindowInterface import InteraceMainwindow
import os
import sys
import time
import win32api
import win32con
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PySide6.QtCore import Qt, QFile, QTextStream
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType

Icons_paths = [
        {"app_name": "YouTube","icon_path": "assets/res/Youtube.png"},
        {"app_name": "Spotify","icon_path": "assets/res/Spotify.png"},
        {"app_name": "Discord","icon_path": "assets/res/Discord.png"},
        {"app_name": "Zoom","icon_path": "assets/res/Zoom.png"},
        {"app_name": "Microsoft Teams","icon_path": "assets/res/Teams.png"},
        {"app_name": "Skype","icon_path": "assets/res/Skype.png"},
        {"app_name": "Telegram Desktop","icon_path": "assets/res/Telegram.png"},
        {"app_name": "WhatsApp","icon_path": "assets/res/Whatsapp.png"},
        {"app_name": "Microsoft Solitaire Collection","icon_path": "assets/res/Solitaire.png"},
        {"app_name": "Default","icon_path": "assets/res/default_app.png"},
    ]

def setup_Icons(app_name, icon_paths):
    default_icon = "assets/res/default_app.png"
    icon_path = default_icon
    
    for idx, icon in enumerate(icon_paths):
        if app_name == icon['app_name']:
            icon_path = icon['icon_path']
            break
    return icon_path

class MainWindow(QMainWindow):
    def __init__(self, choices):
        super().__init__()
        # Configure DPI awareness before UI setup
        self.configure_dpi_awareness()

        self.selectedChoice = None
        self.allchoices = choices
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.ui = InteraceMainwindow()
        self.ui.setupUi(self)

        # Configure ChromeDriver
        self.chrome_service = self.setup_chrome_service()

        # Connect search callback
        self.ui.search_callback = self.handle_youtube_search
        # self.ui.telegram_search_callback = self.handle_telegram_search
        self.move(990, 580)

        for idx, choice in enumerate(choices[0:2]):
            icon = setup_Icons(choice['app_name'], Icons_paths)
            self.add_choice_main(choice['text'], idx, icon, self.on_clicked_choice)
        self.ui.Close_Btn.clicked.connect(self.close)

    def add_choice_main(self, text, id=None, icon_path= None, on_click = None):
        self.ui.add_choice(text, id, icon_path, on_click)

    def on_clicked_choice(self, content, id):
        selected = self.allchoices[id]
        self.selectedChoice = selected
        self.ui.selected_choice = selected  # Store for default query
        
        if selected.get('app_name', '').lower() == 'youtube':
            self.ui.show_search()
            self.ui.search_input.setFocus()
            default_query = selected.get('search_query', '')
            self.ui.search_input.setPlaceholderText(f"e.g. {default_query}" if default_query else "Enter search query...")
        elif selected.get('app_name', '').lower() == 'telegram desktop':
            self.ui.show_search("telegram")
            self.ui.search_input.setFocus()
            self.ui.search_input.setPlaceholderText("Enter Telegram username (e.g., @username)")
        else:
            print(f"Selected: {id},{content}")
            self.close()

    
    def handle_youtube_search(self, query):
        if query is None:  # User cancelled
            return
            
        # Find the YouTube option
        youtube_choice = None
        for choice in self.allchoices:
            if choice.get('app_name', '').lower() == 'youtube':
                youtube_choice = choice
                break
                
        if youtube_choice:
            # Update the choice with the custom query
            youtube_choice['search_query'] = query
            self.selectedChoice = youtube_choice
            self.close()
    #def show_recommendations(self, show):

    # def handle_telegram_search(self, username):
    #     if not username or not username.strip():
    #         QMessageBox.warning(self, "Input Error", "Please enter a Telegram username")
    #         return
        
    #     username = username.lstrip('@')

    #      # Show processing dialog
    #     progress = QMessageBox(self)
    #     progress.setWindowTitle("Searching")
    #     progress.setText(f"Searching for @{username}...")
    #     progress.setStandardButtons(QMessageBox.NoButton)
    #     progress.show()
    #     QApplication.processEvents()
        
    #     try:
    #         user_id = self.get_telegram_user_id(username)
    #         if user_id:
    #             # Process successful result
    #             telegram_choice = next(
    #                 (c for c in self.allchoices 
    #                 if c.get('app_name', '').lower() == 'telegram desktop'),
    #                 None
    #             )
    #             if telegram_choice:
    #                 telegram_choice.update({
    #                     'telegram_user_id': user_id,
    #                     'telegram_username': f"@{username}",
    #                     'app_url': f"https://t.me/{username}"
    #                 })
    #                 self.selectedChoice = telegram_choice
    #     except Exception as e:
    #         QMessageBox.critical(self, "Search Error", f"Failed to search user: {str(e)}")
    #     finally:
    #         progress.close()
    #         self.close()

    # def configure_dpi_awareness(self):
    #     """Multi-layered DPI awareness configuration"""
    #     try:
    #         # Windows API level (works on Windows 10 1607+)
    #         ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
    #     except:
    #         try:
    #             # Fallback for older Windows versions
    #             ctypes.windll.user32.SetProcessDPIAware()
    #         except:
    #             pass
        
    #     # Qt level
    #     os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    #     os.environ["QT_SCALE_FACTOR"] = "1"
    #     os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    #     QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    #     QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    def configure_dpi_awareness(self):
        """Safe DPI settings for Qt 6 (no low-level Windows API)"""
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
        os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)


    def setup_chrome_service(self):
        """Robust ChromeDriver setup with multiple fallbacks"""
        try:
            # Method 1: Automatic management with proper ChromeType detection
            driver_path = ChromeDriverManager(chrome_type=ChromeType.GOOGLE).install()
            service = Service(driver_path)
            # Test the driver
            options = Options()
            options.add_argument("--headless")
            test_driver = webdriver.Chrome(service=service, options=options)
            test_driver.quit()
            return service
        except Exception as e:
            print(f"ChromeDriver auto-install failed: {e}")
            
            # # Method 2: Try common paths
            # common_paths = [
            #     "chromedriver.exe",
            #     os.path.join(os.getenv("ProgramFiles"), "Google", "Chrome", "Application", "chromedriver.exe"),
            #     os.path.join(os.getenv("ProgramFiles(x86)"), "Google", "Chrome", "Application", "chromedriver.exe")
            # ]
            
            # for path in common_paths:
            #     if os.path.exists(path):
            #         try:
            #             service = Service(path)
            #             # Test the driver
            #             options = Options()
            #             options.add_argument("--headless")
            #             test_driver = webdriver.Chrome(service=service, options=options)
            #             test_driver.quit()
            #             return service
            #         except Exception as e:
            #             print(f"ChromeDriver at {path} failed: {e}")
            #             continue
            
            # # Method 3: Try system PATH
            # try:
            #     service = Service()
            #     options = Options()
            #     options.add_argument("--headless")
            #     test_driver = webdriver.Chrome(service=service, options=options)
            #     test_driver.quit()
            #     return service
            # except:
            #     QMessageBox.critical(None, "ChromeDriver Error",
            #         "Could not initialize ChromeDriver.\n\n"
            #         "Please ensure:\n"
            #         "1. Chrome is installed from https://www.google.com/chrome/\n"
            #         "2. Matching ChromeDriver is downloaded from https://chromedriver.chromium.org/\n"
            #         "3. ChromeDriver is placed in your project folder or system PATH")
            #     return None

    def get_telegram_user_id(self, username):
        """Robust Telegram user ID extraction"""
        if not self.chrome_service:
            return None

        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1280,720")
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        driver = None

        try:
            driver = webdriver.Chrome(service=self.chrome_service, options=options)
            driver.get("https://web.telegram.org/")
            
            # Show persistent dialog until user confirms scan
            msg = QMessageBox(self)
            msg.setWindowTitle("Telegram Login Required")
            msg.setText("Please:\n1. Scan the QR code in the browser window\n2. Click OK when logged in")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
            
            # Search for user
            search_box = WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'input[placeholder="Search"]'))
            )
            search_box.clear()
            search_box.send_keys(username)
            
            # Wait for and click user
            user_element = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, f'//div[contains(@class, "ChatInfo") and contains(., "{username}")]'))
            )
            user_element.click()
            
            # Extract ID from URL
            WebDriverWait(driver, 10).until(
                lambda d: "#" in d.current_url
            )
            return driver.current_url.split("#")[-1]
            
        except Exception as e:
            QMessageBox.warning(self, "Search Error", 
                f"Failed to find Telegram user:\n{str(e)}\n\n"
                "Please ensure:\n"
                "1. You're logged in to Telegram Web\n"
                "2. The username exists in your contacts")
            return None
        finally:
            if driver:
                driver.quit()

def launch_window(options):
    # Enable High-DPI scaling
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    """Function to create and return the window instance"""
    # app = QApplication.instance()
    # if not app:
    #     app = QApplication(sys.argv)
    #     # Apply style sheet if needed
    #     style_file = QFile("style.qss")
    #     if style_file.open(QFile.ReadOnly | QFile.Text):
    #         stream = QTextStream(style_file)
    #         app.setStyleSheet(stream.readAll())

    # Create application instance
    app = QApplication(sys.argv)
    
    # Apply styles if available
    if os.path.exists("style.qss"):
        with open("style.qss", "r") as f:
            app.setStyleSheet(f.read())
    
    # Create and show main window
    # window = MainWindow(options)
    # window.show()

    
    window = MainWindow(choices=options)
    window.show()
    # Debug output
    print("Application and window initialized:")
    print(f"App: {app}")
    print(f"Window: {window}")
    return window, app

if __name__ == "__main__":
    app = QApplication(sys.argv)

    options = [
        {'text': 'Watch relaxing video'},
        {'text': 'Watch funny video'}
    ]
    
    window = MainWindow(options)
    window.show()
    app.exec()
    print("Final choice: ", window.selectedChoice)


