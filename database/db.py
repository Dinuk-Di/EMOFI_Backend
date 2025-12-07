# database/db.py
import sqlite3
import os
from typing import List, Tuple
import uuid
import csv
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
# # Navigate up one level and then into assets
# db_path = os.path.join(current_dir, '..', 'assets', 'app.db')

# # Normalize the path (handles the ..)
# DB_PATH = os.path.normpath(db_path)

def get_persistent_db_path():
    # Detect base directory of app.exe or script
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
    else:
        app_dir = os.path.dirname(os.path.abspath(__file__))

    # Use user-specific folder for data storage
    local_appdata = os.getenv('LOCALAPPDATA', app_dir)
    emoify_data_dir = os.path.join(local_appdata, 'Emoify')
    os.makedirs(emoify_data_dir, exist_ok=True)

    # Permanent DB location
    return os.path.join(emoify_data_dir, 'app.db')

DB_PATH = get_persistent_db_path()

def initialize_db():
    if not os.path.exists("assets"):
        os.makedirs("assets")

    conn = sqlite3.connect(DB_PATH)
    create_users_table(conn)
    app_settings(conn)
    recommendation_history(conn)
    emotions(conn)
    apps(conn)
    agent_recommendations(conn)
    return conn

def data_initialization():
    if not os.path.exists("assets"):
        os.makedirs("assets")
    conn = sqlite3.connect(DB_PATH)
    add_emotions(conn)
    add_initial_apps(conn)
    initial_settings()
    return conn

def get_connection():
    if not os.path.exists(DB_PATH):
        initialize_db()
    return sqlite3.connect(DB_PATH)

def create_users_table(conn):
    query = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        phonenumber TEXT,
        birthday TEXT,
        session_id TEXT
    );
    """
    conn.execute(query)
    conn.commit()

def get_user_by_id(user_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return user

def get_user_by_credentials(username: str, password: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()
    conn.close()
    return user

def app_settings(conn):
    query = """
    CREATE TABLE IF NOT EXISTS app_settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        setting_name TEXT NOT NULL,
        setting_value TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );
    """
    conn.execute(query)
    conn.commit()

#convert the value to integer
def get_app_setting(setting_name, default_value):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT setting_value FROM app_settings WHERE setting_name = ?", (setting_name,))
    result = cursor.fetchone()
    conn.close()
    if result:
        try:
            return int(result[0])
        except ValueError:
            return default_value
    return default_value

def initial_settings():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO app_settings (user_id, setting_name, setting_value) VALUES (?, ?, ?)", (1, "theme", "light"))
    cursor.execute("INSERT INTO app_settings (user_id, setting_name, setting_value) VALUES (?, ?, ?)", (1, "systemDisable", "0"))
    cursor.execute("INSERT INTO app_settings (user_id, setting_name, setting_value) VALUES (?, ?, ?)", (1, "recommendationTime", "30"))
    cursor.execute("INSERT INTO app_settings (user_id, setting_name, setting_value) VALUES (?, ?, ?)", (1, "resetTime", "50"))
    cursor.execute("INSERT INTO app_settings (user_id, setting_name, setting_value) VALUES (?, ?, ?)", (1, "appExecuteTime", "60"))
    cursor.execute("INSERT INTO app_settings (user_id, setting_name, setting_value) VALUES (?, ?, ?)", (1, "soundLevel", "2"))
    cursor.execute("INSERT INTO app_settings (user_id, setting_name, setting_value) VALUES (?, ?, ?)", (1, "focusDetection", "0"))
    cursor.execute("INSERT INTO app_settings (user_id, setting_name, setting_value) VALUES (?, ?, ?)", (1, "handDetection", "0"))
    conn.commit()
    conn.close()

def agent_recommendations(conn):
    query = """
    CREATE TABLE IF NOT EXISTS agent_recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        recommendation_type TEXT NOT NULL,
        recommed_app TEXT NOT NULL,
        app_url TEXT,
        search_query TEXT,
        is_local BOOLEAN NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );
    """
    conn.execute(query)
    conn.commit()

def add_agent_recommendations(conn, user_id, recommendation_type, recommed_app, app_url, search_query, is_local):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO agent_recommendations (user_id, recommendation_type, recommed_app, app_url, search_query, is_local)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, recommendation_type, recommed_app, app_url, search_query, is_local))
    print(f"[Info] Added agent recommendation for user {user_id}: {recommed_app}")
    conn.commit()


def recommendation_history(conn):
    query = """
    CREATE TABLE IF NOT EXISTS recommendation_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        emotion TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        selected_previous_recommendation TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );
    """
    conn.execute(query)
    conn.commit()

def recommendation_history_main():
    conn = get_connection()
    query = """
    CREATE TABLE IF NOT EXISTS recommendation_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        emotion TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        selected_previous_recommendation TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    );
    """
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM apps ORDER BY id DESC LIMIT 10")
    all_data = cursor.fetchall()
    conn.commit()

def emotions(conn):
    query = """
    CREATE TABLE IF NOT EXISTS emotions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        emotion TEXT NOT NULL,
        is_positive BOOLEAN NOT NULL
    );
    """
    conn.execute(query)
    conn.commit()
    



# Add emotion function
def add_emotions(conn):
    emotions_data = [
        ("Angry", False),
        ("Boring", False),
        ("Disgust", False),
        ("Fear", False),
        ("Happy", True),
        ("Neutral", True),
        ("Sad", False),
        ("Stress", False),
        ("Surprise", True),
    ]

    cursor = conn.cursor()
    for emotion, is_positive in emotions_data:
        cursor.execute("INSERT INTO emotions (emotion, is_positive) VALUES (?, ?)", (emotion, is_positive))
    conn.commit()

def add_app_data(conn, user_id, category, app_name, app_id, app_url, path, is_local, app_type):
    cursor = conn.cursor()
    cursor.execute("""
            INSERT INTO apps (user_id, category, app_name, app_id, app_url, path, is_local, app_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, category, app_name, app_id, app_url, path, is_local, app_type))
    conn.commit()

# get app_type, app_name and app_id only
def get_app_data(conn, app_name):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT app_type, app_name, app_id, path
        FROM apps
        WHERE app_name = ? AND is_local = 1
    """, (app_name,))
    row = cursor.fetchone()

    if row:
        return {
            "app_type": row[0],
            "name": row[1],
            "app_id": row[2],
            "path": row[3]
        }
    else:
        return {
            "app_type": "Unknown",
            "name": app_name,
            "app_id": None,
            "path": None
        }


def delete_app_data(conn, app_id: int):
    """
    Deletes an app entry from the database.

    :param conn: SQLite connection
    :param app_id: ID of the app to delete
    """
    cursor = conn.cursor()
    cursor.execute("DELETE FROM apps WHERE id = ?", (app_id,))
    conn.commit()


def apps(conn):
    query = """
    CREATE TABLE IF NOT EXISTS apps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        category TEXT NOT NULL CHECK(category IN (
            'Songs', 'Entertainment', 'SocialMedia', 'Games', 'Communication', 'Help', 'Other')),
        app_name TEXT NOT NULL,
        app_id TEXT,
        app_url TEXT,
        path TEXT,
        is_local BOOLEAN NOT NULL,
        app_type TEXT NOT NULL CHECK(app_type IN ('uwp', 'classic','web')),
        FOREIGN KEY (user_id) REFERENCES users (id)
    );
    """
    conn.execute(query)
    conn.commit()

def add_initial_apps(conn):
    """
    Adds initial apps to the database.
    """
    initial_apps = [
        (1, 'Games', 'Subway Surfers', None, 'https://poki.com/en/g/subway-surfers', None, False, 'web'),
        (1, 'Games', 'Brain Test', None, 'https://poki.com/en/g/brain-test-tricky-puzzles', None, False, 'web'),
        (1, 'Games', 'Bike Game', None, 'https://poki.com/en/g/stunt-bike-extreme', None, False, 'web'),
        (1, 'Entertainment', 'YouTube', None, 'https://www.youtube.com', None, False, 'web'),
        (1, 'Entertainment', 'Films', None, 'https://myflixerz.to/', None, False, 'web'),
        (1, 'Songs', 'Spotify', None, 'https://open.spotify.com/', None, False, 'web'),
        (1, 'Songs', 'SoundCloud', None, 'https://soundcloud.com/', None, False, 'web'),
        (1, 'SocialMedia', 'WhatsApp', None, 'https://web.whatsapp.com/', None, False, 'web'),
        (1, 'SocialMedia', 'Facebook', None, 'https://www.facebook.com/', None, False, 'web'),
        (1, 'Help', 'ChatGPT', None, 'https://chatgpt.com/', None, False, 'web'),
        (1, 'Help', 'Perplexity', None, 'https://www.perplexity.ai/', None, False, 'web')

    ]

    cursor = conn.cursor()
    cursor.executemany("""
        INSERT INTO apps (user_id, category, app_name, app_id, app_url, path, is_local, app_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, initial_apps)
    conn.commit()

def get_apps(conn) -> List[Tuple]:
    """
    Fetches most recent 10 apps from the database .

    """
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM apps ORDER BY id DESC LIMIT 10")
    all_data = cursor.fetchall()
    print("[App Data]:", all_data)
    # get name, category and path only
    filtered_data = [(row[2], row[3], row[4], row[6]) for row in all_data]
    return filtered_data