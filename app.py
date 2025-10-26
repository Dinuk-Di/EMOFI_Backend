from flask import Flask, request, jsonify
from flask_cors import CORS
from database.database import init_db, save_UserData, get_user_by_username, get_user_settings, set_user_settings
from database.db import get_connection, get_user_by_id
from old_utils.state import app_state, pickle_save, pickle_load
import time, threading
from main import start_app
from customtkinter import CTk
from ui.register import AppRegister
import sqlite3

database_file = r'assets\app.db'

app = Flask(__name__)
CORS(app)

database = {
    "state": {
        "system_status": "Operational",
        "emotion_state": "Neutral",
        "last_response_time": "18 mins ago",
    },
    "recent_records": {
        "recommendation": "Relaxing music",
        "action": "Youtube"
    },
    "apps": [
        {
            "name": "Spotify",
            "icon": "https://example.com/spotify-icon.png",
            "path": "https://www.spotify.com",
            "description": "Music streaming service",
            "type": "entertainment",
            "mode": "online",
            "isAccessGiven": True,
            "isAvailable": True,
        },
        {
            "name": "Youtube",
            "icon": "https://example.com/youtube-icon.png",
            "path": "https://www.youtube.com",
            "description": "Video streaming service",
            "type": "entertainment",
            "mode": "online",
            "isAccessGiven": True,
            "isAvailable": True,
        },
        {
            "name": "Discord",
            "icon": "https://example.com/discord-icon.png",
            "path": "https://www.discord.com",
            "description": "Voice, video, and text communication service",
            "type": "communication",
            "mode": "online",
            "isAccessGiven": False,
            "isAvailable": True,
        }
    ],
    "recommendation": [
        {
            "recommendation": "Relaxing music",
            "apps": [
                {
                    "app": "Youtube",
                    "iconPath": "/com/example/emoify_javafx/icons/youtube.png"
                },
                {
                    "app": "Spotify",
                    "iconPath": "/com/example/emoify_javafx/icons/Spotify.png"
                },
                {
                    "app": "Discord",
                    "iconPath": "/com/example/emoify_javafx/icons/Discord.png"
                }
            ],
        },
        {
            "recommendation": "Play games",
            "apps": [
                {
                    "app": "Solitaire",
                    "iconPath": "/com/example/emoify_javafx/icons/Solitaire.png"
                },
                {
                    "app": "Delta Force",
                    "iconPath": "/com/example/emoify_javafx/icons/Delta_Force.png"
                },
                {
                    "app": "Breathedge",
                    "iconPath": "/com/example/emoify_javafx/icons/Breathedge.png"
                }
            ],
        },
        {
            "recommendation": "Watch movies",
            "apps": [
                {
                    "app": "Movie Player",
                    "iconPath": "/com/example/emoify_javafx/icons/default_app.png"
                },
                {
                    "app": "Movie Player 2",
                    "iconPath": "/com/example/emoify_javafx/icons/default_app.png"
                },
                {
                    "app": "Movie Player 3",
                    "iconPath": "/com/example/emoify_javafx/icons/default_app.png"
                }
            ],
        }
    ],
    "showRecommendation": {
        "show": True
    },
    "selectedApp": {
        "name": "Spotify"
    }
}

@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/api/saveUserData', methods=['POST'])
def save_user_data():
    data = request.json
    userName = data.get('userName')
    password = data.get('password')
    phoneNumber = data.get('phoneNumber')

    if not all([userName, password, phoneNumber]):
        return jsonify({"error": "Missing required fields"}), 400

    save_UserData(userName, password, phoneNumber)
    return jsonify({"message": "User data saved successfully"}), 201

@app.route('/api/getUserData', methods=['GET'])
def get_user_data():
    userName = request.args.get('userName')
    if not userName:
        return jsonify({"error": "userName parameter is required"}), 400
    
    user = get_user_by_username(userName)
    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({"userName": user[0], "phoneNumber": user[2]}), 200

@app.route('/api/state', methods=['GET'])
def get_state():
    return jsonify(database["state"])

@app.route('/api/recentRecords', methods=['GET'])
def get_recent_records():
    
    return jsonify(database["recent_records"])

@app.route('/api/apps', methods=['GET'])
def get_apps():
    return jsonify(database["apps"])

@app.route('/api/recommendation', methods=['GET'])
def get_recommendation():
    return jsonify(database["recommendation"])

@app.route('/api/showRecommendation', methods=['GET'])
def get_show_recommendation():
    return jsonify(database["showRecommendation"])

@app.route('/api/setShowRecommendation', methods=['POST'])
def set_show_recommendation():
    data = request.json
    show_recommendation = data.get('showRecommendation')
    if show_recommendation is not None:
        database["showRecommendation"]["show"] = show_recommendation
        return jsonify({"message": "Show recommendation updated successfully"}), 200
    return jsonify({"error": "Invalid data"}), 400

@app.route('/api/setSelectedApp', methods=['POST'])
def set_selected_app():
    data = request.json
    selected_app = data.get('selectedApp')
    if selected_app is not None:
        database["selectedApp"]["name"] = selected_app
        return jsonify({"message": "Selected app updated successfully"}), 200
    return jsonify({"error": "Invalid data"}), 400

#new database endpoints
@app.route('/api/getLogin', methods=['GET'])
def get_login():
    user = get_user_by_id(1)
    if user:
        return jsonify({"message": "Login successful", "user": user}), 200
    return jsonify({"error": "User not found"}), 404

#new app adder endpoint
@app.route('/api/addApp', methods=['POST'])
def add_app():
    data = request.json
    user_name = data.get('user')

    root = CTk()
    AppRegister(root, user_name)
    root.mainloop()
    return jsonify({"message": "App registration window opened"}), 200

#settings endpoint
@app.route('/api/getSettings', methods=['GET'])
def get_settings():
    userID = 1
    if not userID:
        return jsonify({"error": "userID parameter is required"}), 400

    settings = initial_user_settings(userID)
    if not settings:
        return jsonify({"error": "No settings found for user"}), 404

    return jsonify({"settings": settings}), 200


@app.route('/api/editSettings', methods=['POST'])
def edit_settings():
    data = request.json
    userID = data.get('userID')
    settings = data.get('settings')

    if not all([userID, settings]):
        return jsonify({"error": "Missing required fields"}), 400

    results = []

    conn = sqlite3.connect(database_file, timeout=5)  # Wait up to 5 sec
    cursor = conn.cursor()

    try:
        for setting in settings:
            if not all(key in setting for key in ['name', 'value']):
                results.append({"error": f"Invalid setting format: {setting}"})
                continue

            cursor.execute(
                    "UPDATE app_settings SET setting_value=? WHERE user_id=? AND setting_name=?",
                    (setting['value'], userID, setting['name'])
                )

            results.append({
                "setting": setting['name'],
                "status": "updated",
                "new_value": setting['value']
            })

        conn.commit()

    except sqlite3.OperationalError as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()
        
    return jsonify({
        "message": "Batch settings update complete"
    }), 200

#state management endpoints
@app.route('/api/getExecutedState', methods=['GET'])
def get_executed_state():
    return jsonify({"show": get_execution_state()})

@app.route('/api/setExecutedState', methods=['POST'])
def set_executed_state():
    data = request.json
    executed = data.get('executed')
    recommendation = data.get('recommendation')
    recommendedApp = data.get('recommendedApp')
    if executed is not None:
        app_state.executedApp = executed
        app_state.selectedApp = recommendedApp
        app_state.selectedRecommendation = recommendation
        pickle_save()
        return jsonify({"message": "Execution state updated successfully"}), 200
    return jsonify({"error": "Invalid data"}), 400

@app.route('/api/getRecommendationOptionsState', methods=['GET'])
def get_recommendation_options():
    return jsonify(get_recommendation_options())

@app.route('/api/setSelectedAppState', methods=['POST'])
def set_selectedAppState():
    data = request.json
    selected_app = data.get('selectedApp')
    if selected_app is not None:
        app_state.selectedApp = selected_app
        pickle_save()
        app_state.reset()
        return jsonify({"message": "Selected app updated successfully"}), 200
    return jsonify({"error": "Invalid data"}), 400

@app.route('/api/setStateInit', methods=['POST'])
def set_state_init():
    app_state.reset()
    pickle_save()
    return jsonify({"message": "App state reset successfully"}), 200

# app ui status update
@app.route('/api/stateUI', methods=['GET'])
def get_state_UI():
    return jsonify(get_system_status())

def start_flask():
    app.run(debug=True, port=5000)

def initialize_app():
    flask_thread = threading.Thread(target=start_flask)
    flask_thread.daemon = True
    flask_thread.start()

    start_app()

def get_execution_state():
    app_state = pickle_load()
    return app_state.executed

def get_recommendation_options():
    app_state = pickle_load()
    return app_state.recommendations

def get_system_status():

    app_state = pickle_load()

    average_emotion = app_state.averageEmotion if app_state.averageEmotion else "Neutral"

    return {
        "system_status": database["state"]["system_status"],
        "emotion_state": average_emotion,
        "last_response_time": database["state"]["last_response_time"]
    }

def initial_user_settings(userID):
    user_settings = get_user_settings(userID)
    if not user_settings:
        set_user_settings(userID, "theme", "light")
        set_user_settings(userID, "systemDisable", "false")
        set_user_settings(userID, "recommendationTime", "10")
        set_user_settings(userID, "restTime", "25")
        set_user_settings(userID, "appExecuteTime", "10")
        set_user_settings(userID, "soundLevel", "Mid")
        return get_user_settings(userID)
    return user_settings

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
