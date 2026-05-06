from collections import defaultdict
from datetime import datetime

memory_store = defaultdict(list)

def save_message(session_id: str, role: str, content: str):
    memory_store[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    if len(memory_store[session_id]) > 20:
        memory_store[session_id] = memory_store[session_id][-20:]

def get_history(session_id: str) -> list:
    return memory_store[session_id]

def get_history_as_text(session_id: str) -> str:
    history = memory_store[session_id]
    if not history:
        return "No previous conversation."
    text = "Previous conversation:\n"
    for msg in history:
        text += f"{msg['role'].upper()}: {msg['content']}\n"
    return text

def clear_memory(session_id: str):
    memory_store[session_id] = []