import customtkinter as ctk
from PIL import Image
import os
from ui.pages.home import HomePage
from ui.pages.setting import SettingsPage
from ui.pages.chatbot import ChatbotPage
from ui.pages.logs import LogsPage
from core.controller import AppController
from queue import Queue


class Dashboard:
    def __init__(self, root, username="User"):
        self.root = root
        self.log_queue = Queue()
        self.root.title("EMOFI - Dashboard")
        self.position_bottom_right()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.controller = AppController(log_queue=self.log_queue)

        # App Icon
        icon_path = os.path.join(os.path.dirname(__file__), "..", "assets","res", "Icon.ico")
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)

        # Main Frame
        main_frame = ctk.CTkFrame(self.root, fg_color="#1a1a1a")
        main_frame.pack(fill="both", expand=True)

        # Sidebar Frame
        sidebar = ctk.CTkFrame(main_frame, width=100, fg_color="#2b2b2b")
        sidebar.pack(side="left", fill="y")

        # Content Frame
        self.content_frame = ctk.CTkFrame(main_frame, fg_color="#1a1a1a")
        self.content_frame.pack(side="right", fill="both", expand=True)

        self.icons = {
            "home": self.load_icon("home.png", size=(25, 25)),
            "settings": self.load_icon("settings.png", size=(25, 25)),
            "chatbot": self.load_icon("bot.png", size=(25, 25)),
            "logs": self.load_icon("log.png", size=(25, 25)),
        }


        # Sidebar Buttons
        self.active_button = None
        self.pages = {}

        self.home_btn = self.create_sidebar_button(sidebar, "Home", self.icons["home"], lambda: self.show_page("home"))
        self.settings_btn = self.create_sidebar_button(sidebar, "Settings", self.icons["settings"], lambda: self.show_page("settings"))
        self.chatbot_btn = self.create_sidebar_button(sidebar, "Chatbot", self.icons["chatbot"], lambda: self.show_page("chatbot"))
        self.logs_btn = self.create_sidebar_button(sidebar, "Logs", self.icons["logs"], lambda: self.show_page("logs"))

        self.home_btn.pack(pady=(20, 5))
        self.settings_btn.pack(pady=5)
        self.chatbot_btn.pack(pady=5)
        self.logs_btn.pack(pady=5)

        # Bottom Control Bar
        control_bar = ctk.CTkFrame(sidebar, fg_color="#2b2b2b")
        control_bar.pack(side="bottom", pady=20)

        play_icon = self.create_icon_button(control_bar, "▶", "#4CAF50", command=lambda: self.controller.start())
        close_icon = self.create_icon_button(control_bar, "✖", "#F44336", command=lambda: self.controller.stop())
        exercise_icon = self.create_icon_button(control_bar, "🏋️", "#2196F3", command=lambda: self.controller.exercise())


        # icons should be vertically aligned
        play_icon.pack(side="top", pady=5)
        close_icon.pack(side="top", pady=5)
        exercise_icon.pack(side="top", pady=5)
    
        # Initialize Pages
        self.pages["home"] = HomePage(self.content_frame)
        self.pages["settings"] = SettingsPage(self.content_frame)
        self.pages["chatbot"] = ChatbotPage(self.content_frame)
        self.pages["logs"] = LogsPage(self.content_frame, self.log_queue)

        # Show Home by default
        self.show_page("home")

    def position_bottom_right(self):
        self.root.update_idletasks()
        width = 400
        height = 450
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = screen_width - 170
        y = screen_height - height - 20
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.resizable(False, False)

    def load_icon(self, filename, size=(40, 40)):
        path = os.path.join(os.path.dirname(__file__), "..", "assets/res", filename)
        if os.path.exists(path):
            img = Image.open(path)
            return ctk.CTkImage(light_image=img, dark_image=img, size=size)
        return None

    
    def create_sidebar_button(self, parent, text, image, command):
        btn = ctk.CTkButton(
            parent,
            text=text,
            image=image,
            width=50,
            height=40,
            corner_radius=8,
            font=("Arial", 8), 
            fg_color="#3a3a3a",
            hover_color="#4a4a4a",
            compound="top",
            command=lambda: [self.set_active(btn), command()]
        )
        return btn


    def create_icon_button(self, parent, symbol, hover_color, command):
        btn = ctk.CTkButton(parent, text=symbol, width=35, height=35, corner_radius=6, fg_color="#333",
                            hover_color=hover_color, command=command)
        return btn

    def set_active(self, button):
        if self.active_button:
            self.active_button.configure(fg_color="#3a3a3a")
        button.configure(fg_color="#0e3aa9")
        self.active_button = button

    def show_page(self, page_name):
        for widget in self.content_frame.winfo_children():
            widget.pack_forget()
        self.pages[page_name].frame.pack(fill="both", expand=True)

    def close_app(self):
        self.root.destroy()


def open_dashboard(username):
    root = ctk.CTk()
    Dashboard(root, username)
    root.mainloop()
