import customtkinter as ctk
from customtkinter import CTkImage
from PIL import Image, ImageTk
import os
import requests
import json
from dotenv import load_dotenv
import subprocess

load_dotenv()
QWEN_API_KEY = os.getenv("QWEN_API_KEY")

initial_emotion = "Stress"
chat_history = []

PROMPT_TEMPLATE = """
You are a friendly, empathetic chatbot. Your main goal is to improve the user's mood and make them feel positive.
The user currently feels: {emotion}.
If the user emotion is negative, you will respond with a positive message to help lift their mood.
Start a conversation with the user, asking how they are doing and offering support.
Give jokes or positive affirmations, motivational quotes, or suggestions to cheer them up.
Respond in a friendly, positive, and conversational tone.
Conversation so far:
{history}
User: {user_input}
Bot:
"""

def query_huggingface(prompt):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {QWEN_API_KEY}",
            "Content-Type": "application/json"
        },
        data=json.dumps({
            "model": "qwen/qwen-2.5-72b-instruct:free",
            "messages": [{"role": "user", "content": prompt}]
        })
    )
    return response.json()

def get_chatbot_response(user_input):
    global chat_history
    history_text = "\n".join(chat_history[-5:])
    prompt = PROMPT_TEMPLATE.format(emotion=initial_emotion, history=history_text, user_input=user_input)

    result = query_huggingface(prompt)
    try:
        bot_response = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        bot_response = "Sorry, I couldn't generate a response right now. Please try again!"

    chat_history.append(f"User: {user_input}")
    chat_history.append(f"Bot: {bot_response}")
    return bot_response


class ChatbotPage:
    def __init__(self, parent):
        self.frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.frame.pack_propagate(False)

        self.help_bot_window = None        

        # Paths
        base_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "res")
        user_img_path = os.path.join(base_path, "user.png")
        bot_img_path = os.path.join(base_path, "Icon1.jpg")
        send_icon_path = os.path.join(base_path, "send.png")

        # Load icons
        self.user_icon = CTkImage(light_image=Image.open(user_img_path), size=(25, 25))
        self.bot_icon = CTkImage(light_image=Image.open(bot_img_path), size=(25, 25))
        self.send_icon = CTkImage(light_image=Image.open(send_icon_path), size=(20, 20))

        # ✅ Check Button for Help Bot
        check_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        check_frame.pack(side="top", fill="x", pady=(5, 0))

        self.help_var = ctk.BooleanVar(value=False)
        self.help_checkbox = ctk.CTkCheckBox(
            check_frame,
            text="Enable Help Bot",
            variable=self.help_var,
            command=self.toggle_help_bot
        )
        self.help_checkbox.pack(side="left", padx=10, pady=5)

        # Main chat area
        main_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        main_frame.pack(fill="both", expand=True)

        # Scrollable Chat Area
        self.chatbox_canvas = ctk.CTkCanvas(main_frame, bg="#1E1E1E", highlightthickness=0)
        self.chatbox_canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ctk.CTkScrollbar(main_frame, orientation="vertical", command=self.chatbox_canvas.yview)
        scrollbar.pack(side="right", fill="y")

        self.chatbox_frame = ctk.CTkFrame(self.chatbox_canvas, fg_color="transparent")
        self.chatbox_frame.bind("<Configure>", lambda e: self.chatbox_canvas.configure(scrollregion=self.chatbox_canvas.bbox("all")))
        self.chatbox_canvas.create_window((0, 0), window=self.chatbox_frame, anchor="nw")
        self.chatbox_canvas.configure(yscrollcommand=scrollbar.set)

        # Input Area
        input_frame = ctk.CTkFrame(self.frame, fg_color="#2C2F33")
        input_frame.pack(side="bottom", fill="x", pady=8)

        self.entry = ctk.CTkEntry(input_frame, width=270, placeholder_text="Type a message...")
        self.entry.grid(row=0, column=0, padx=(10, 5))
        self.entry.bind("<Return>", self.send_message)

        send_btn = ctk.CTkButton(
            input_frame,
            text="",
            image=self.send_icon,
            width=30,
            height=30,
            corner_radius=6,
            fg_color="#3a3a3a",
            hover_color="#4a90e2",
            command=self.send_message
        )
        send_btn.grid(row=0, column=1, padx=2)

        # Initial bot message
        self.add_message("Hi! I heard you're feeling a bit down. Let's chat and make your day better! 😊", "bot")

    def toggle_help_bot(self):
        if self.help_var.get():
            # Open help_bot.py window
            if self.help_bot_window is None:
                self.help_bot_window = subprocess.Popen(
                    ["python", "help_bot.py"],
                    cwd=os.path.dirname(os.path.dirname(__file__)))
        else:
            # Close the help bot window if open
            if self.help_bot_window:
                self.help_bot_window.terminate()
                self.help_bot_window = None


    def send_message(self, event=None):
        user_input = self.entry.get().strip()
        if user_input == "":
            return

        self.add_message(user_input, "user")
        self.entry.delete(0, "end")

        bot_response = get_chatbot_response(user_input)
        self.add_message(bot_response, "bot")

    def add_message(self, message, sender="bot"):
        message_frame = ctk.CTkFrame(self.chatbox_frame, fg_color="transparent")
        message_frame.pack(fill="x", pady=4, padx=6, anchor="w" if sender == "bot" else "e")

        icon = self.bot_icon if sender == "bot" else self.user_icon
        icon_label = ctk.CTkLabel(message_frame, image=icon, text="")
        icon_label.pack(side="left" if sender == "bot" else "right", padx=4)

        bubble_color = "#2A2D32" if sender == "bot" else "#1F6AA5"
        text_label = ctk.CTkLabel(
            message_frame,
            text=message,
            wraplength=260,  # ✅ Adjusted for small width
            fg_color=bubble_color,
            corner_radius=12,
            justify="left",
            padx=8,
            pady=6
        )
        text_label.pack(side="left" if sender == "bot" else "right")

        self.chatbox_canvas.update_idletasks()
        self.chatbox_canvas.yview_moveto(1.0)
