import customtkinter as ctk
from tkinter import messagebox
from database.db import get_connection
# from ui.dashboard_ import open_dashboard
from ui.dashboard import open_dashboard
import os
import sys
import subprocess
from PIL import Image,ImageTk

class LoginWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("EMOFI - Login")
        self.position_bottom_right()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

         # ✅ Set App Icon (for window and taskbar)
        icon_path = os.path.join(os.path.dirname(__file__), "..", "assets/res", "Icon.jpg")
        img_icon = None
        if os.path.exists(icon_path):
            pil_image = Image.open(icon_path)
            tk_icon = ImageTk.PhotoImage(pil_image)
            self.root.iconphoto(False, tk_icon)
            img_icon = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(50, 50))

        # Main Frame
        frame = ctk.CTkFrame(root, corner_radius=15)
        frame.pack(fill="both", expand=True, padx=15, pady=15)

        # App Icon & Name
        ctk.CTkLabel(frame, text="EMOFI", font=("Arial", 28, "bold")).pack(pady=(20, 10))
        if os.path.exists(icon_path):
            ctk.CTkLabel(frame, image=img_icon, text="").pack(pady=(0, 15))

        # space for better alignment
        ctk.CTkLabel(frame, text="", height=50).pack()

        # Username
        # ctk.CTkLabel(frame, text="Username", font=("Arial", 14)).pack(pady=(10, 2))
        self.username_entry = ctk.CTkEntry(frame, width=220, height=35, placeholder_text="Enter username")
        self.username_entry.pack(pady=(0, 15))


        # Buttons
        ctk.CTkButton(frame, text="Login", width=220, height=40, corner_radius=8, command=self.login).pack(pady=(0, 8))
        ctk.CTkButton(frame, text="Register", width=220, height=40, corner_radius=8, fg_color="gray", command=self.register).pack()

        # Optional: Chatbot Button
        # ctk.CTkButton(frame, text="Open Chatbot", width=220, height=35, corner_radius=8, fg_color="#4CAF50", command=self.open_chatbot).pack(pady=15)

    def position_bottom_right(self):
        self.root.update_idletasks()
        width = 400
        height = 450
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = screen_width - width  - 110
        y = screen_height - height - 220
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.resizable(False, False)

    def login(self):
        username = self.username_entry.get()
        

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users")
        users = cursor.fetchall()
        conn.close()

        if len(users) == 1:
            self.root.destroy()
            open_dashboard(users[0][1])
            return

        for user in users:
            if user[1] == username:
                self.root.destroy()
                open_dashboard(username)
                return

        messagebox.showerror("Login Failed", "Invalid username or password.")

    def register(self):
        from ui.register import RegisterWindow
        self.root.destroy()
        root = ctk.CTk()
        RegisterWindow(root)
        root.mainloop()

    def open_chatbot(self):
        chatbot_file = os.path.join(os.path.dirname(__file__), "chatbot.py")
        python_exec = sys.executable
        subprocess.Popen([python_exec, chatbot_file])
