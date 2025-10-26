import customtkinter as ctk
from tkinter import messagebox
from database.db import data_initialization
from ui.app_register import AppRegister
import uuid
import os
from PIL import Image, ImageTk


class RegisterWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Emofi - Registration")
        self.position_bottom_right()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # ✅ Set App Icon (window and app bar)
        icon_path = os.path.join(os.path.dirname(__file__), "..", "assets/res", "Icon1.jpg")

        app_icon = os.path.join(os.path.dirname(__file__), "..", "assets", "res", "Icon.ico")

        # For Windows taskbar and title bar
        if os.path.exists(app_icon):
            self.root.iconbitmap(app_icon)
        
        # For Tkinter image usage
        if os.path.exists(icon_path):
            pil_image = Image.open(icon_path)
            tk_icon = ImageTk.PhotoImage(pil_image)
            self.root.iconphoto(False, tk_icon)
            img_icon = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(60, 60))
        else:
            img_icon = None

        # ✅ Main Frame
        frame = ctk.CTkFrame(root, corner_radius=15)
        frame.pack(padx=20, pady=20, fill="both", expand=True)

        # ✅ App Icon
        if img_icon:
            ctk.CTkLabel(frame, image=img_icon, text="").pack(pady=(15, 5))

        # ✅ App Name
        ctk.CTkLabel(frame, text="Emofi", font=("Arial", 28, "bold")).pack(pady=(0, 1))


        # ✅ Quote
        ctk.CTkLabel(frame, text="Register to make your work easy", font=("Arial", 12), text_color="#a9a9a9").pack(pady=(0, 5))

        # ✅ Username
        ctk.CTkLabel(frame, text="Username").pack()
        self.username_entry = ctk.CTkEntry(frame, width=250, height=35, placeholder_text="Enter username")
        self.username_entry.pack(pady=(5, 10))

        # ✅ Phone Number
        ctk.CTkLabel(frame, text="Phone Number").pack()
        self.phone_entry = ctk.CTkEntry(frame, width=250, height=35, placeholder_text="Enter phone number")
        self.phone_entry.pack(pady=(5, 10))

        # ✅ Birthday
        ctk.CTkLabel(frame, text="Birthday (YYYY-MM-DD)").pack()
        self.birthday_entry = ctk.CTkEntry(frame, width=250, height=35, placeholder_text="YYYY-MM-DD")
        self.birthday_entry.pack(pady=(5, 15))

        # ✅ Next Button
        ctk.CTkButton(frame, text="Next", width=250, height=40, corner_radius=8,
                      command=self.register_and_continue).pack(pady=15)

    def position_bottom_right(self):
        self.root.update_idletasks()
        width = 400
        height = 500
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = screen_width - width  - 110
        y = screen_height - height - 220
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.resizable(False, False)

    def register_and_continue(self):
        username = self.username_entry.get()
        phone = self.phone_entry.get()
        birthday = self.birthday_entry.get()
        session_id = str(uuid.uuid4())

        if not username or not phone or not birthday:
            messagebox.showwarning("Warning", "Please fill in all fields.")
            return

        conn = data_initialization()
        try:
            conn.execute(
                "INSERT INTO users (username, phonenumber, birthday, session_id) VALUES (?, ?, ?, ?)",
                (username, phone, birthday, session_id)
            )
            conn.commit()
            messagebox.showinfo("Success", "Registration successful!")

            self.root.destroy()
            root = ctk.CTk()
            AppRegister(root, username)
            root.mainloop()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            conn.close()
