import uuid
import requests

CHATBOT_URL = "http://127.0.0.1:9000/chat"
session_id = str(uuid.uuid4())

print("Chatbot (Option 1) - type 'exit' to quit\n")

while True:
    msg = input("You: ").strip()
    if msg.lower() in {"exit", "quit"}:
        break

    try:
        # Detect "open <n>" commands and print "Downloading......" immediately,
        # before the HTTP request blocks waiting for ingestion to complete.
        is_open_cmd = msg.lower().startswith("open ")
        if is_open_cmd:
            print("\nBot: Downloading......\n", flush=True)

        # Increased timeout to survive slow upstream arXiv retry windows
        r = requests.post(
            CHATBOT_URL,
            json={"session_id": session_id, "message": msg},
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
        reply = data["reply"]

        if is_open_cmd:
            # "Downloading......" was already printed — just print the rest
            print(reply, "\n")
        else:
            print("\nBot:", reply, "\n")

    except requests.exceptions.ReadTimeout:
        print("\nBot: arXiv/backend is taking too long right now (timeout). Try again in a bit.\n")

    except requests.exceptions.RequestException as e:
        print(f"\nBot: request failed: {e}\n")