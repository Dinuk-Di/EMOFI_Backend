import customtkinter as ctk
from queue import Queue

class LogsPage:
    def __init__(self, parent, log_queue: Queue):
        self.frame = ctk.CTkFrame(parent, fg_color="#1a1a1a")
        self.log_queue = log_queue
        
        # Title
        title_label = ctk.CTkLabel(self.frame, text="📜 Live Logs", font=("Arial", 20, "bold"))
        title_label.pack(anchor="w", padx=15, pady=(15, 5))
        

        # Scrollable Log Box
        self.log_textbox = ctk.CTkTextbox(self.frame, height=400, width=500, wrap="word")
        self.log_textbox.pack(padx=15, pady=10, fill="both", expand=True)
        self.log_textbox.configure(state="disabled")

        # Store reference to log queue
        

        # Start updating logs
        self.update_logs()

    def update_logs(self):
        # Pull logs from queue and display
        while not self.log_queue.empty():
            message = self.log_queue.get_nowait()
            self.log_textbox.configure(state="normal")
            self.log_textbox.insert("end", f"{message}\n")
            self.log_textbox.see("end")  # Auto-scroll
            self.log_textbox.configure(state="disabled")

        # Schedule next update
        self.frame.after(500, self.update_logs)
