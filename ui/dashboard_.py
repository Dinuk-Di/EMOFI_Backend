import customtkinter as ctk
from core.controller import AppController
from database.db import initialize_db
from queue import Queue

controller = None

def open_dashboard(username):
    global controller
    log_queue = Queue()
    ctk.set_appearance_mode("dark")  # "light", "dark", or "system"
    ctk.set_default_color_theme("blue")  # Options: blue, green, dark-blue

    app = ctk.CTk()
    app.title(f"Welcome, {username}")
    app.geometry("700x600")
    app.resizable(False, False)

    # Header Frame
    header_frame = ctk.CTkFrame(app, corner_radius=10)
    header_frame.pack(padx=20, pady=20, fill="x")

    title_label = ctk.CTkLabel(header_frame, text="PC Monitor Dashboard", font=("Arial", 20, "bold"))
    title_label.pack(pady=5)

    user_label = ctk.CTkLabel(header_frame, text=f"Logged in as: {username}", font=("Arial", 14))
    user_label.pack()

    # Control Panel
    control_frame = ctk.CTkFrame(app, corner_radius=10)
    control_frame.pack(padx=20, pady=10, fill="x")

    ctk.CTkButton(control_frame, text="Start Monitoring", command=lambda: controller.start(), width=200).pack(pady=10)
    ctk.CTkButton(control_frame, text="Stop Monitoring", command=lambda: controller.stop(), width=200).pack(pady=10)
    ctk.CTkButton(control_frame, text="Exit", command=app.destroy, width=200, fg_color="red", hover_color="#cc0000").pack(pady=10)

    # Footer or Extra Info
    footer_frame = ctk.CTkFrame(app, corner_radius=10)
    footer_frame.pack(padx=20, pady=10, fill="x")

    ctk.CTkLabel(footer_frame, text="Status: Idle", font=("Arial", 12)).pack(pady=5)

     # Log Frame (for detection outputs)
    log_frame = ctk.CTkFrame(app, corner_radius=10)
    log_frame.pack(padx=20, pady=(0, 10), fill="both", expand=True)

    log_label = ctk.CTkLabel(log_frame, text="Live Logs:", font=("Arial", 14, "bold"))
    log_label.pack(anchor="w", padx=10, pady=(5, 0))

    log_textbox = ctk.CTkTextbox(log_frame, height=450)
    log_textbox.pack(padx=10, pady=5, fill="both", expand=True)
    log_textbox.configure(state="disabled")

    controller = AppController(log_queue=log_queue)

    def update_logs():
        # print("[GUI] Checking log queue...")  # debug
        while not log_queue.empty():
            message = log_queue.get_nowait()
            log_textbox.configure(state="normal")
            log_textbox.insert("end", f"{message}\n")
            log_textbox.see("end")  # auto-scroll
            log_textbox.configure(state="disabled")
        app.after(500, update_logs)  # call again after 500ms

    update_logs()  # start the loop

    app.mainloop()



