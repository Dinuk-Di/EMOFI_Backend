import sqlite3

database = r'assets\app.db'

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
    conn = sqlite3.connect(database)
    c = conn.cursor()
    c.execute(
        "INSERT INTO Users (userName, password, phoneNumber) VALUES (?, ?, ?)", 
        (userName, password, phoneNumber)
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