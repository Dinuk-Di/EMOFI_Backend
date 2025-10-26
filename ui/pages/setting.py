import customtkinter as ctk

class SettingsPage:
    def __init__(self, parent):
        self.frame = ctk.CTkFrame(parent, fg_color="#1a1a1a")
        label = ctk.CTkLabel(self.frame, text="Settings Page", font=("Arial", 20, "bold"))
        label.pack(pady=50)
