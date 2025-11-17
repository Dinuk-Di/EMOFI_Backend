import sys
import subprocess
from tkinter import messagebox
from database.db import get_connection
from database.db import initialize_db
from ui.register import RegisterWindow
from ui.dashboard import open_dashboard

def is_do_not_disturb_enabled():
    """Check if Windows Do Not Disturb (Focus Assist) is enabled."""
    try:
        result = subprocess.run(
            ['powershell', '-Command',
             'Get-ItemProperty -Path "HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Notifications\\Settings" | Select-Object -ExpandProperty NOC_GLOBAL_SETTING_TOASTS_ENABLED'],  # Fixed typo: SETTING -> SETTING
            capture_output=True, text=True
        )
        value = result.stdout.strip()
        return value == '0'  # 0 = Do Not Disturb enabled
    except Exception as e:
        print("Failed to check DND:", e)
        return False

def disable_do_not_disturb():
    """Disable Windows Do Not Disturb (Focus Assist)."""
    try:
        subprocess.run([
            'powershell', '-Command',
            'Set-ItemProperty -Path "HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Notifications\\Settings" -Name NOC_GLOBAL_SETTING_TOASTS_ENABLED -Value 1'  # Name corrected here too
        ])
        print("Do Not Disturb disabled.")
    except Exception as e:
        print("Failed to disable DND:", e)

def main():
    # Check and handle Do Not Disturb
    if is_do_not_disturb_enabled():
        root = CTk()
        root.withdraw()  # Hide the window while prompting
        response = messagebox.askokcancel(
            "Do Not Disturb Enabled",
            "Do Not Disturb is currently enabled on your system.\n"
            "This may block app notifications.\n\n"
            "Would you like to disable it?"
        )
        root.destroy()

        if response:
            disable_do_not_disturb()
        else:
            sys.exit("App closed due to Do Not Disturb being enabled.")

    # Connect to database and try to fetch user info
    conn = get_connection()
    cursor = conn.cursor()
    initialize_db()
    cursor.execute("SELECT username, session_id FROM users WHERE id = 1")
    user_data = cursor.fetchone()

    username, session_id = (None, None)

    if user_data:
        username, session_id = user_data
        print(f"Username: {username}, Session ID: {session_id}")
    else:
        print("User not found.")

    # Launch appropriate window based on user data
    if username and session_id:
        
        open_dashboard(username)

        # app.mainloop()
    else:
        # Open registration window if user not found
        root = CTk()
        RegisterWindow(root)
        root.mainloop()

if __name__ == "__main__":
    main()
