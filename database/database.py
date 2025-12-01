import sqlite3
import os

from flask import jsonify

#database = r'assets\app.db'

current_dir = os.path.dirname(os.path.abspath(__file__))  # C:/project/database

# Navigate up one level and then into assets
db_path = os.path.join(current_dir, '..', 'assets', 'app.db')

# Normalize the path (handles the ..)
database = os.path.normpath(db_path)

def init_db():
    conn = sqlite3.connect(database)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS Users
                 (userName TEXT PRIMARY KEY, password TEXT, phoneNumber TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS SystemSettings
                 (mode TEXT PRIMARY KEY, value TEXT)''')
    conn.commit()
    conn.close()

def save_UserData(userName, password, phoneNumber):
    user_id = 1
    conn = sqlite3.connect(database)
    c = conn.cursor()
    c.execute(
        "INSERT INTO Users (id, userName, password, phoneNumber) VALUES (?, ?, ?, ?)", 
        (user_id, userName, password, phoneNumber)
    )
    conn.commit()
    conn.close()

def get_user_by_username(userName):
    """Get a single user by their userName (ID)"""
    conn = sqlite3.connect(database)
    c = conn.cursor()
    c.execute("SELECT * FROM Users WHERE userName = ?", (userName,))
    user = c.fetchone()
    conn.close()
    return user

def get_user_settings(userID):
    conn = sqlite3.connect(database)
    c = conn.cursor()
    c.execute("SELECT * FROM app_settings WHERE user_id = ?", (userID,))
    settings = c.fetchall()
    conn.close()
    return settings

def set_user_settings(userID, setting_name, setting_value):
    conn = sqlite3.connect(database)
    c = conn.cursor()
    c.execute("UPDATE app_settings SET setting_value = ? WHERE user_id = ? AND setting_name = ?",
              (setting_value, userID, setting_name))
    conn.commit()
    conn.close()
    return True

def set_user_settings_initial():
    user_id = 1  # Assuming you want to set initial settings for user with ID 1
    default_settings = {
        "theme": "light",
        "systemDisable": "false",
        "recommendationTime": "5",
        "restTime": "10",
        "appExecuteTime": "10",
        "soundLevel": "Mid"
    }
    conn = sqlite3.connect(database)
    c = conn.cursor()
    for setting_name, setting_value in default_settings.items():
        c.execute("INSERT INTO app_settings (user_id, setting_name, setting_value) VALUES (?, ?, ?)",
                  (user_id, setting_name, setting_value))
    conn.commit()
    conn.close()

def get_all_apps():
    user_id = 1  # Assuming you want to fetch apps for user with ID 1
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT app_name as name, path, category, is_local as isLocal, is_available as isAvailable
        FROM apps 
        WHERE user_id = ?
    """, (user_id,))
    
    apps = cursor.fetchall()
    conn.close()

    return apps