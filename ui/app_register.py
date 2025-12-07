import customtkinter as ctk
from tkinter import filedialog, messagebox
import winreg
from database.db import get_connection, add_app_data
import os
from PIL import Image, ImageDraw
from ui.dashboard import open_dashboard
import subprocess

ASSET_PATH = "assets/res"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class AppRegister:
    def __init__(self, root, username):
        self.root = root
        self.username = username
        
        self.root.title("Emofi - App Registration")
        self.position_bottom_right()

        # App Icon
        icon_path = os.path.join(os.path.dirname(__file__), "..", "assets","res", "Icon.ico")
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)
     

        self.categories = ["Songs", "Entertainment", "SocialMedia", "Games", "Communication", "Help", "Other"]
        self.category_data = {}

        self.main_frame = ctk.CTkScrollableFrame(self.root, corner_radius=10)
        self.main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        self.build_ui()

    # def position_bottom_right(self):
    #     self.root.update_idletasks()
    #     width = 400
    #     height = 450
    #     screen_width = self.root.winfo_screenwidth()
    #     screen_height = self.root.winfo_screenheight()
    #     x = screen_width - width  - 110
    #     y = screen_height - height - 220
    #     self.root.geometry(f"{width}x{height}+{x}+{y}")
    #     self.root.resizable(False, False)

    def position_bottom_right(self):
        self.root.update_idletasks()
        self.root.attributes('-topmost', True)
        width = 400
        height = 450
      
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        # x = Screen Width - Window Width - Padding (e.g., 20px margin)
        x = screen_width - width - 120
        
        # y = Screen Height - Window Height - Padding (e.g., 50px for taskbar)
        y = screen_height - height - 170
        
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.resizable(False, False)

    def build_ui(self):
        # ✅ Header
        ctk.CTkLabel(self.main_frame, text="Register Your Applications", font=("Arial", 20, "bold")).pack(pady=4)

        # ✅ Add small description text
        ctk.CTkLabel(
            self.main_frame,
            text="Add the apps which are installed in your system",
            font=("Arial", 12),
            text_color="gray"
        ).pack(pady=(0, 5))

        for category in self.categories:
            self.add_category_block(category)

        # ✅ Submit Button
        ctk.CTkButton(self.main_frame, text="Next", command=self.submit).pack(pady=20)


    def add_category_block(self, category):
        block = ctk.CTkFrame(self.main_frame, border_width=1, corner_radius=10)
        block.pack(pady=10, fill="x", padx=5)

        header_frame = ctk.CTkFrame(block, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(header_frame, text=category, font=("Arial", 16, "bold")).pack(side="left")

        # Load Add Icon image
        add_icon_path = os.path.join(ASSET_PATH, "plus.png")
        add_icon_img = None
        if os.path.exists(add_icon_path):
            add_icon_img = ctk.CTkImage(light_image=Image.open(add_icon_path), size=(25, 25))

        # Replace "+" button
        add_button = ctk.CTkButton(
            header_frame,
            text="",  # Remove text
            image=add_icon_img,
            width=40,
            height=40,
            fg_color="transparent",
            hover_color="#333333",
            command=lambda c=category: self.open_add_app_popup(c, block)
        )
        add_button.pack(side="left", padx=(10, 0))

        # Horizontal container for app icons
        app_icon_frame = ctk.CTkScrollableFrame(
            block, fg_color="transparent", orientation="horizontal", height=120
        )
        app_icon_frame.pack(fill="x", padx=10, pady=(5, 10))

        self.category_data[category] = {
            "apps": [],
            "app_icon_frame": app_icon_frame,
            "app_widgets": []
        }
    
    def load_app_icon(self, app_name):
        filename = f"{app_name.lower().replace(' ', '')}.png"
        path = os.path.join(ASSET_PATH, filename)
        fallback_path = os.path.join(ASSET_PATH, "Other.png")

        try:
            img = Image.open(path)
        except FileNotFoundError:
            img = Image.open(fallback_path)

        img = img.resize((50, 50)).convert("RGBA")
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, img.size[0], img.size[1]), fill=255)
        img.putalpha(mask)
        return ctk.CTkImage(light_image=img, size=(50, 50))

    def confirm_delete_app(self, category, index, parent_frame):
        confirm = messagebox.askyesno("Delete App", "Are you sure you want to remove this app?")
        if confirm:
            del self.category_data[category]["apps"][index]
            self.update_app_list(category, parent_frame)


    def update_app_list(self, category, parent_frame):
        # Clear existing widgets
        for widget in self.category_data[category]["app_widgets"]:
            widget.destroy()
        self.category_data[category]["app_widgets"] = []

        frame = self.category_data[category]["app_icon_frame"]

        for index, (name, path_val,) in enumerate(self.category_data[category]["apps"]):
            icon_image = self.load_app_icon(name)

            app_frame = ctk.CTkFrame(frame, fg_color="transparent", width=80, height=100)
            app_frame.pack(side="left", padx=8, pady=5)

            # Icon container to position delete button over it
            icon_container = ctk.CTkFrame(app_frame, fg_color="transparent", width=60, height=60)
            icon_container.pack()
            icon_container.pack_propagate(False)

            icon_label = ctk.CTkLabel(icon_container, text="", image=icon_image)
            icon_label.image = icon_image
            icon_label.pack()

            delete_icon_path = os.path.join(ASSET_PATH, "delete.png")
            delete_icon_img = None
            if os.path.exists(delete_icon_path):
                delete_icon_img = ctk.CTkImage(light_image=Image.open(delete_icon_path), size=(20,20))

            # Replace delete button
            delete_btn = ctk.CTkButton(
                icon_container,
                text="",
                image=delete_icon_img,
                width=24,
                height=24,
                fg_color="transparent",
                hover_color="#ff4444", 
                corner_radius=1,
                command=lambda idx=index: self.confirm_delete_app(category, idx, parent_frame)
            )
            delete_btn.place(relx=1.0, rely=1.0, anchor="se", x=-2, y=-2)

            name_label = ctk.CTkLabel(app_frame, text=name, font=("Arial", 10), wraplength=70)
            name_label.pack(pady=(4, 0))

            self.category_data[category]["app_widgets"].append(app_frame)


    def open_add_app_popup(self, category, parent_frame):
        popup = ctk.CTkToplevel(self.root)
        popup.title("Add Application")
        popup_width, popup_height = 350, 300
        screen_width = popup.winfo_screenwidth()
        screen_height = popup.winfo_screenheight()
        x = (screen_width // 2) - (popup_width // 2)
        y = (screen_height // 2) - (popup_height // 2)
        popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")
        popup.resizable(False, False)

        ctk.CTkLabel(popup, text=f"Add App to '{category}'", font=("Arial", 16, "bold")).pack(pady=10)

        # Name entry
        ctk.CTkLabel(popup, text="App Name:").pack(anchor="w", padx=20)
        name_frame = ctk.CTkFrame(popup)
        name_frame.pack(pady=5)
        name_entry = ctk.CTkEntry(name_frame, width=240)
        name_entry.pack(side="left", padx=(0, 5))

        # Location entry
        ctk.CTkLabel(popup, text="App Location / ID:").pack(anchor="w", padx=20)
        location_frame = ctk.CTkFrame(popup)
        location_frame.pack(pady=5)
        location_entry = ctk.CTkEntry(location_frame, width=240)
        location_entry.pack(side="left", padx=(0, 5))

        def browse_file():
            path = filedialog.askopenfilename()
            if path:
                location_entry.delete(0, "end")
                location_entry.insert(0, path)
                popup.app_type = "classic"
                popup.app_id = None
                popup.app_path = path

        browse_btn = ctk.CTkButton(location_frame, text="📁", width=40, command=browse_file)
        browse_btn.pack(side="left")

        # ---------- Search Logic ----------
        def find_executable_in_folder(folder_path):
            if not os.path.isdir(folder_path):
                return None
            exe_files = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(".exe"):
                        exe_files.append(os.path.join(root, file))
            return exe_files[0] if exe_files else None

        def get_classic_installed_programs():
            uninstall_keys = [
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
                r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"
            ]
            results = []
            for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                for subkey in uninstall_keys:
                    try:
                        with winreg.OpenKey(root, subkey) as key:
                            for i in range(0, winreg.QueryInfoKey(key)[0]):
                                try:
                                    subkey_name = winreg.EnumKey(key, i)
                                    with winreg.OpenKey(key, subkey_name) as app_key:
                                        name = winreg.QueryValueEx(app_key, "DisplayName")[0]
                                        location = ""
                                        try:
                                            location = winreg.QueryValueEx(app_key, "InstallLocation")[0]
                                        except:
                                            pass
                                        if name:
                                            results.append((name, location))
                                except:
                                    continue
                    except:
                        continue
            return results

        def get_installed_uwp_apps():
            try:
                command = [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy", "Bypass",
                    "-Command",
                    "Get-StartApps | ForEach-Object { \"$($_.Name)|$($_.AppID)\" }"
                ]
                output = subprocess.check_output(command, text=True)
                apps = []
                for line in output.strip().splitlines():
                    if "|" in line:
                        name, app_id = line.split("|", 1)
                        apps.append((name.strip(), app_id.strip()))
                return apps
            except:
                return []

        def search_installed_apps():
            name = name_entry.get().lower().strip()
            if not name:
                messagebox.showwarning("Missing", "Enter app name before searching")
                return

            # Try UWP
            uwp_matches = [app for app in get_installed_uwp_apps() if name in app[0].lower()]
            if uwp_matches:
                app_name, app_id = uwp_matches[0]
                location_entry.delete(0, "end")
                location_entry.insert(0, app_id)
                popup.app_type = "uwp"
                popup.app_id = app_id
                popup.app_path = None
                messagebox.showinfo("Found", f"UWP App Found: {app_name}")
                return

            # Try Classic
            classic_matches = [app for app in get_classic_installed_programs() if name in app[0].lower()]
            if classic_matches:
                app_name, install_location = classic_matches[0]
                exe_path = find_executable_in_folder(install_location) or install_location
                location_entry.delete(0, "end")
                location_entry.insert(0, exe_path)
                popup.app_type = "classic"
                popup.app_id = None
                popup.app_path = exe_path
                messagebox.showinfo("Found", f"Classic App Found: {app_name}")
                return

            messagebox.showwarning("Not Found", "App not found, please browse manually.")

        search_btn = ctk.CTkButton(name_frame, text="🔍", width=40, command=search_installed_apps)
        search_btn.pack(side="left")

        # ---------- Save ----------
        def save_app():
            app_name = name_entry.get().strip()
            if not app_name:
                messagebox.showwarning("Missing", "Please enter app name")
                return

            # If user browsed manually and didn't search
            if not hasattr(popup, "app_type"):
                path_val = location_entry.get().strip()
                if not path_val:
                    messagebox.showwarning("Missing", "Please select app location or search")
                    return
                popup.app_type = "classic"
                popup.app_id = None
                popup.app_path = path_val

            # Store in DB
            self.category_data[category]["apps"].append((app_name, popup.app_path or popup.app_id))
            conn = get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM users WHERE username = ?", (self.username,))
                user_id = cursor.fetchone()[0]

                add_app_data(
                    conn=conn,
                    user_id=user_id,
                    category=category,
                    app_name=app_name,
                    app_id=popup.app_id,
                    app_url=None,
                    path=popup.app_path,
                    is_local=True,
                    app_type=popup.app_type
                )

                messagebox.showinfo("Success", "App added successfully.")
                popup.destroy()
                self.update_app_list(category, parent_frame)
            except Exception as e:
                messagebox.showerror("Database Error", f"Failed to add app: {e}")
                print(f"[DB Error] {e}")

        ctk.CTkButton(popup, text="Add", command=save_app).pack(pady=15)



    def submit(self):
        collected_data = {}
        for category in self.categories:
            apps = self.category_data[category]["apps"]
            collected_data[category] = {"apps": apps}      
        print("Collected Data:", collected_data)
        
        messagebox.showinfo("Submitted", "Your preferences have been saved!")
        self.root.destroy()
        open_dashboard(self.username)



if __name__ == "__main__":
    root = ctk.CTk()
    username = "test_user"  # Replace with actual username logic
    app_register = AppRegister(root, "selani")
    root.mainloop()
