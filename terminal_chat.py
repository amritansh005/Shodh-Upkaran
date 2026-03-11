import re
import uuid
import requests

CHATBOT_URL = "http://127.0.0.1:9000/chat"
session_id = str(uuid.uuid4())


def strip_markdown(text: str) -> str:
    """Remove common Markdown formatting for plain-terminal display."""
    # Fenced code blocks: ```lang\n...\n``` — must come before inline-code pass
    text = re.sub(r'```[^\n]*\n(.*?)```', r'\1', text, flags=re.DOTALL)
    # Bold+italic: ***text***
    text = re.sub(r'\*{3}(.+?)\*{3}', r'\1', text, flags=re.DOTALL)
    # Bold: **text** or __text__
    text = re.sub(r'\*{2}(.+?)\*{2}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'_{2}(.+?)_{2}',   r'\1', text, flags=re.DOTALL)
    # Italic: *text* or _text_  (skip lone list-bullet * / _ at line start)
    text = re.sub(r'(?<!\s)\*(.+?)\*(?!\s)', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'(?<!\s)_(.+?)_(?!\s)',   r'\1', text, flags=re.DOTALL)
    # Inline code: `text`
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Headers: ### Heading → Heading
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Blockquotes: > text → text
    text = re.sub(r'^\s*>\s?', '', text, flags=re.MULTILINE)
    # Horizontal rules
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Images (before links so the leading ! is consumed): ![alt](url) → alt
    text = re.sub(r'!\[(.+?)\]\(.*?\)', r'\1', text)
    # Links: [text](url) → text
    text = re.sub(r'\[(.+?)\]\(.*?\)', r'\1', text)
    # Strikethrough: ~~text~~ → text
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    return text


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
        reply = strip_markdown(data["reply"])

        if is_open_cmd:
            # "Downloading......" was already printed — just print the rest
            print(reply, "\n")
        else:
            print("\nBot:", reply, "\n")

    except requests.exceptions.ReadTimeout:
        print("\nBot: arXiv/backend is taking too long right now (timeout). Try again in a bit.\n")

    except requests.exceptions.RequestException as e:
        print(f"\nBot: request failed: {e}\n")