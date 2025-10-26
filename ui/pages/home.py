import customtkinter as ctk

class HomePage:
    def __init__(self, parent):
        self.frame = ctk.CTkFrame(parent, fg_color="#1a1a1a")
        label = ctk.CTkLabel(self.frame, text="Welcome to EMOFI Home", font=("Arial", 20, "bold"))
        label.pack(pady=50)
