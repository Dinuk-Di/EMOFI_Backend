import customtkinter as ctk
import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_API_KEY")

# Initial Negative Emotion
initial_emotion = "Stress"
chat_history = []

# Prompt Template
PROMPT_TEMPLATE = """
You are a friendly, empathetic chatbot. Your main goal is to improve the user's mood and make them feel positive.
The user currently feels: {emotion}.
If the user emotion is negative, you will respond with a positive message to help lift their mood.
start a conversation with the user, asking how they are doing and offering support. Like ask them about their day, hobbies, or interests.
Give some jokes or positive affirmations to cheer them up. Also give some motivational quotes or suggestions to help them feel better.
You will always respond in a friendly, positive, and encouraging manner.
Always respond with kindness, optimism, and a conversational tone.
Conversation so far:
{history}
User: {user_input}
Bot:
"""

# HuggingFace Inference API call
def query_huggingface(prompt):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
                "Authorization": f"Bearer {QWEN_API_KEY}",
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": "qwen/qwen-2.5-72b-instruct:free",
                "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
                ]
            }))
    return response.json()

# Generate chatbot response
def get_chatbot_response(user_input):
    global chat_history

    # Build prompt with conversation history
    history_text = "\n".join(chat_history[-5:])  # Keep last 5 exchanges
    prompt = PROMPT_TEMPLATE.format(emotion=initial_emotion, history=history_text, user_input=user_input)

    # API Request
    result = query_huggingface(prompt)

    # Extract response from OpenRouter structure
    try:
        bot_response = result["choices"][0]["message"]["content"]
        print(f"Bot Response: {bot_response}")
    except (KeyError, IndexError):
        bot_response = "Sorry, I couldn't generate a response right now. Please try again!"

    # Update chat history
    chat_history.append(f"User: {user_input}")
    chat_history.append(f"Bot: {bot_response}")

    return bot_response