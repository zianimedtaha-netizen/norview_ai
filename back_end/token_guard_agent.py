import os
import time
import threading
from datetime import datetime
from collections import defaultdict
import pytz
from dotenv import load_dotenv

load_dotenv()

PACIFIC_TZ = pytz.timezone("US/Pacific")
_lock = threading.Lock()

# ── MODELS ──
CHAT_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemma-3-4b-it",
]

# ── GLOBAL USAGE ──
_state = {
    "total_tokens": 0,
    "total_limit": 900000,
    "model_index": 0,
    "last_reset": datetime.now(PACIFIC_TZ).date(),
    "chat_requests": 0,
}

# ── PER-SESSION TOKEN TRACKING ──
_session_tokens: dict = defaultdict(lambda: {"tokens": 0, "reset_at": time.time() + 3600})
_session_lock = threading.Lock()

SESSION_TOKEN_LIMIT = 10000  # per hour


def _check_daily_reset():
    today = datetime.now(PACIFIC_TZ).date()
    if today > _state["last_reset"]:
        with _lock:
            if today > _state["last_reset"]:
                _state["total_tokens"] = 0
                _state["model_index"] = 0
                _state["chat_requests"] = 0
                _state["last_reset"] = today
                print("[TokenGuard] Daily reset — back to gemini-2.5-flash")


def _get_model_index() -> int:
    pct = (_state["total_tokens"] / _state["total_limit"]) * 100 if _state["total_limit"] else 0
    if pct >= 90 and _state["model_index"] == 0:
        with _lock:
            if _state["model_index"] == 0:
                _state["model_index"] = 1
                print("[TokenGuard] Usage >90% → gemini-2.5-flash-lite")
    return _state["model_index"]


def check_session_limit(session_id: str) -> bool:
    """Returns True if session is within limit, False if exceeded."""
    with _session_lock:
        entry = _session_tokens[session_id]
        if time.time() > entry["reset_at"]:
            entry["tokens"] = 0
            entry["reset_at"] = time.time() + 3600
        return entry["tokens"] < SESSION_TOKEN_LIMIT


def update_session_tokens(session_id: str, tokens: int):
    with _session_lock:
        _session_tokens[session_id]["tokens"] += tokens


def get_chat_model() -> str:
    _check_daily_reset()
    idx = _get_model_index()
    return CHAT_MODELS[min(idx, len(CHAT_MODELS) - 1)]


def update_chat_usage(tokens: int, session_id: str = "default"):
    with _lock:
        _state["total_tokens"] += tokens
        _state["chat_requests"] += 1
    update_session_tokens(session_id, tokens)


def handle_quota_error() -> bool:
    with _lock:
        if _state["model_index"] < len(CHAT_MODELS) - 1:
            _state["model_index"] += 1
            print(f"[TokenGuard] Quota error → {CHAT_MODELS[_state['model_index']]}")
            return True
    return False


def check_rate_limit():
    """Enforce minimum spacing between requests (12s)."""
    pass  # handled per-IP in main.py


def safe_generate(prompt: str, session_id: str = "default") -> str:
    """
    Generate response with full protection:
    - Session token limit check
    - Retry up to 2 times
    - Max 2 model switches
    - Failsafe message if all models fail
    """
    import google.generativeai as genai

    _check_daily_reset()

    if not check_session_limit(session_id):
        return "You have reached the usage limit. Please try again later."

    switches = 0
    max_switches = 2

    for attempt in range(3):
        try:
            model_name = get_chat_model()
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and hasattr(response, "text") and response.text:
                try:
                    tokens = response.usage_metadata.total_token_count
                    update_chat_usage(tokens, session_id)
                except Exception:
                    pass
                return response.text
        except Exception as e:
            err = str(e)
            is_quota = "429" in err or "quota" in err.lower() or "404" in err or "not found" in err.lower()
            if is_quota and switches < max_switches:
                switched = handle_quota_error()
                if switched:
                    switches += 1
                    time.sleep(3)
                    continue
            print(f"[TokenGuard] Gemini error attempt {attempt+1}: {e}")
            if attempt < 2:
                time.sleep(2)
                continue

    return "System temporarily under heavy load. Please try again shortly."


def get_usage_status() -> dict:
    _check_daily_reset()
    pct = round((_state["total_tokens"] / _state["total_limit"]) * 100, 2) if _state["total_limit"] else 0
    return {
        "current_model": CHAT_MODELS[min(_state["model_index"], len(CHAT_MODELS) - 1)],
        "total_tokens_used": _state["total_tokens"],
        "total_limit": _state["total_limit"],
        "percentage": pct,
        "chat_requests": _state["chat_requests"],
        "status": "normal" if pct < 60 else "warning" if pct < 90 else "critical",
    }