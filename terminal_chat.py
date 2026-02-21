import uuid
import requests

CHATBOT_URL = "http://127.0.0.1:9000/chat"
session_id = str(uuid.uuid4())

print("Chatbot (Option 1) - type 'exit' to quit\n")

while True:
    msg = input("You: ").strip()
    if msg.lower() in {"exit", "quit"}:
        break

    r = requests.post(CHATBOT_URL, json={"session_id": session_id, "message": msg}, timeout=120)
    r.raise_for_status()
    data = r.json()
    print("\nBot:", data["reply"], "\n")
