import sys
import os
import time
import logging
import threading
import sqlite3
from collections import defaultdict

logging.getLogger("uvicorn.access").addFilter(
    lambda record: "/api/status" not in record.getMessage()
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from memory_agent import save_message, get_history_as_text
from token_guard_agent import safe_generate, get_usage_status, check_session_limit
from ticket_agent import create_ticket

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY or API_KEY == "put_your_real_api_key_here":
    print("WARNING: GEMINI_API_KEY is missing or invalid in .env")
else:
    genai.configure(api_key=API_KEY)

app = FastAPI(title="NorView AI Intelligence System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── STARTUP ──
@app.on_event("startup")
async def startup_event():
    try:
        from knowledge_agent import get_sentence_model
        get_sentence_model()
    except Exception as e:
        print(f"[Startup] Model preload failed: {e}")
    _init_ticket_db()
    try:
        from sentiments_agent import init_db
        init_db()
    except Exception as e:
        print(f"[Startup] Feedback DB init failed: {e}")

# ── TICKET DATABASE (SQLite with retry) ──
_ticket_db_lock = threading.Lock()
TICKET_DB = "tickets.db"

def _init_ticket_db():
    for attempt in range(3):
        try:
            conn = sqlite3.connect(TICKET_DB, timeout=5)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.commit()
            conn.close()
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < 2:
                time.sleep(0.5)
                continue
            print(f"[TicketDB] Init failed: {e}")

def _get_last_ticket_time(session_id: str) -> float:
    for attempt in range(3):
        try:
            with _ticket_db_lock:
                conn = sqlite3.connect(TICKET_DB, timeout=5)
                cur = conn.cursor()
                cur.execute(
                    "SELECT created_at FROM tickets WHERE session_id=? ORDER BY created_at DESC LIMIT 1",
                    (session_id,)
                )
                row = cur.fetchone()
                conn.close()
                return row[0] if row else 0.0
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < 2:
                time.sleep(0.3)
                continue
            return 0.0
    return 0.0

def _store_ticket_record(session_id: str, message: str):
    for attempt in range(3):
        try:
            with _ticket_db_lock:
                conn = sqlite3.connect(TICKET_DB, timeout=5)
                conn.execute(
                    "INSERT INTO tickets (session_id, message, created_at) VALUES (?, ?, ?)",
                    (session_id, message, time.time())
                )
                conn.commit()
                conn.close()
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < 2:
                time.sleep(0.3)
                continue
            print(f"[TicketDB] Store failed: {e}")

TICKET_COOLDOWN = 120  # 2 minutes

# ── RATE LIMITER (25 req/min per IP) ──
RATE_LIMIT = 25
RATE_WINDOW = 60
_rate_lock = threading.Lock()
_request_counts: dict = defaultdict(list)

def check_ip_rate_limit(client_ip: str):
    now = time.time()
    with _rate_lock:
        _request_counts[client_ip] = [
            t for t in _request_counts[client_ip] if now - t < RATE_WINDOW
        ]
        if len(_request_counts[client_ip]) >= RATE_LIMIT:
            raise HTTPException(status_code=429, detail="Too many requests. Please wait before sending again.")
        _request_counts[client_ip].append(now)

# ── IN-MEMORY CACHE (5 min TTL) ──
_cache: dict = {}
_cache_lock = threading.Lock()
CACHE_TTL = 300

def cache_get(key: str):
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (time.time() - entry["ts"]) < CACHE_TTL:
            return entry["value"]
        if entry:
            del _cache[key]
    return None

def cache_set(key: str, value):
    with _cache_lock:
        _cache[key] = {"value": value, "ts": time.time()}

# ── SAFE ROUTE TASK ──
def safe_route_task(message: str, session_id: str) -> dict:
    try:
        from coordinator_agent import route_task
        result = route_task(message, session_id)
        if not isinstance(result, dict):
            return {"agent_result": str(result) if result else "", "knowledge": "", "needs_support": False}
        return {
            "agent_result": result.get("agent_result", "") or "",
            "knowledge":    result.get("knowledge", "") or "",
            "needs_support": bool(result.get("needs_support", False))
        }
    except Exception as e:
        print(f"[route_task] failed: {e}")
        return {"agent_result": "", "knowledge": "", "needs_support": False}

# ── YES/NO KEYWORD DETECTION ──
YES_WORDS = {"yes","oui","نعم","ah","yep","yeah","yup","ايه","آه","واه","ewa","si","awa"}
NO_WORDS  = {"no","non","لا","la","nope","nah","لأ"}

def is_yes(text: str) -> bool:
    return bool(set(text.lower().strip().split()) & YES_WORDS)

def is_no(text: str) -> bool:
    return bool(set(text.lower().strip().split()) & NO_WORDS)

# ── SESSION TICKET STATE ──
ticket_pending: dict = {}
ticket_offered: dict = {}
ticket_context: dict = {}
ticket_sent:    dict = {}

# ── FILLER CLEANER ──
FILLERS = [
    "As an AI language model,","As an AI,","I want to clarify that",
    "It's important to note that","Certainly!","Absolutely!","Of course!",
    "Great question!","I hope this helps!","Please don't hesitate to ask",
    "Feel free to ask","I'm here to help!","As NorView,",
]

def clean_answer(text: str) -> str:
    for f in FILLERS:
        text = text.replace(f, "").strip()
    while "  " in text:
        text = text.replace("  ", " ")
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()

# ── BG TASKS ──
def bg_save_message(session_id, role, content):
    try:
        save_message(session_id, role, content)
    except Exception as e:
        print(f"[BG] save_message failed: {e}")

def bg_store_feedback(input_text, detected_emotion, confidence,
                      all_scores, student_id, star_rating, written_feedback):
    try:
        from sentiments_agent import store_feedback
        store_feedback(
            input_text=input_text,
            detected_emotion=detected_emotion,
            confidence=confidence,
            all_scores=all_scores,
            student_id=student_id,
            star_rating=star_rating,
            written_feedback=written_feedback
        )
    except Exception as e:
        print(f"[BG] store_feedback failed: {e}")

# ── MODELS ──
class PromptRequest(BaseModel):
    prompt: str
    session_id: str = "default"

class TicketRequest(BaseModel):
    name: str
    email: str
    subject: str
    description: str
    session_id: str = "default"

class FeedbackRequest(BaseModel):
    rating: int
    feedback: str = ""
    session_id: str = "default"

# ─────────────────────────────────────────
@app.post("/api/chat")
async def chat_endpoint(request: PromptRequest, req: Request, background_tasks: BackgroundTasks):
    if not API_KEY or API_KEY == "put_your_real_api_key_here":
        raise HTTPException(status_code=500, detail="Gemini API key is not configured.")

    check_ip_rate_limit(req.client.host)

    session = request.session_id
    prompt  = request.prompt.strip()

    # Session token limit
    if not check_session_limit(session):
        return {"answer": "You have reached the usage limit. Please try again later.", "needs_support": False}

    try:
        history = get_history_as_text(session) or ""
        background_tasks.add_task(bg_save_message, session, "student", prompt)

        # ── TICKET PENDING: yes/no only ──
        if ticket_pending.get(session) and not ticket_sent.get(session):
            if is_yes(prompt):
                context = ticket_context.get(session, prompt)
                last_ticket = _get_last_ticket_time(session)
                if time.time() - last_ticket < TICKET_COOLDOWN:
                    ticket_pending[session] = False
                    answer = "You already created a ticket recently. Please wait before creating another."
                    background_tasks.add_task(bg_save_message, session, "nor", answer)
                    return {"answer": answer, "needs_support": False}
                try:
                    create_ticket(context, history, session)
                    _store_ticket_record(session, context)
                    ticket_sent[session] = True
                except Exception as e:
                    print(f"[Ticket] create failed: {e}")
                ticket_pending[session] = False
                answer = "Your support ticket has been created successfully. Our team will contact you soon! 😊"
                background_tasks.add_task(bg_save_message, session, "nor", answer)
                return {"answer": answer, "needs_support": True}

            elif is_no(prompt):
                ticket_pending[session] = False
                answer = "No problem! Let me know if there's anything else I can help you with."
                background_tasks.add_task(bg_save_message, session, "nor", answer)
                return {"answer": answer, "needs_support": False}

        # ── CACHE CHECK ──
        ck = f"{session}:{prompt[:80].lower()}"
        cached = cache_get(ck)
        if cached:
            return cached

        # ── ROUTE TASK ──
        route_result  = safe_route_task(prompt, session)
        agent_result  = route_result["agent_result"]
        knowledge     = route_result["knowledge"]
        needs_support = route_result["needs_support"]

        # ── DIRECT TICKET (no AI) ──
        if needs_support and not ticket_sent.get(session):
            last_ticket = _get_last_ticket_time(session)
            if time.time() - last_ticket < TICKET_COOLDOWN:
                answer = "You already created a ticket recently. Please wait before creating another."
                background_tasks.add_task(bg_save_message, session, "nor", answer)
                return {"answer": answer, "needs_support": False}

            if not ticket_offered.get(session):
                ticket_offered[session] = True
                ticket_context[session] = prompt
                ticket_pending[session] = True
                # Ask once — response generated below will include the question
            # Fall through to generate AI response that asks the question

        mood_context  = f"Student mood detected: {agent_result}\n" if agent_result else ""
        knowledge_ctx = f"\nRELEVANT KNOWLEDGE:\n{knowledge}" if knowledge else ""
        already_sent  = ticket_sent.get(session, False)

        support_instruction = ""
        if needs_support and ticket_pending.get(session) and not already_sent:
            support_instruction = (
                "IMPORTANT: The student needs human help. "
                "At the end of your response, warmly ask ONCE in their language "
                "if they would like you to create a support ticket. 1 sentence only."
            )

        engineered_prompt = f"""You are NorView, an AI assistant exclusively for EMSI students.
You ONLY answer questions related to EMSI, student life, courses, administration, and school topics.
You are NOT a general AI assistant.
If a student asks something unrelated to EMSI, politely redirect them.

CRITICAL LANGUAGE RULE: Detect the language and respond in THAT EXACT SAME LANGUAGE.
- French → respond in French
- Arabic → respond in Arabic  
- Darija → respond in Darija
- English → respond in English
- Mixed → match their mix

<user_request>
{prompt}
</user_request>

{mood_context}{knowledge_ctx}

CONVERSATION HISTORY:
{history}

{support_instruction}

SYSTEM PROMPT: You are NorView helping EMSI students. Consider these 12 points silently:
1. Determine the core question or goal.
2. Consider constraints (beginner student, real-time response).
3. Detect edge cases or ambiguous inputs.
4. Choose the clearest format (short sentence, bullets, example).
5. Ensure privacy — never reveal personal info.
6. Use concrete examples when helpful.
7. Check assumptions about tools and environment.
8. Suggest better alternatives if they exist.
9. Keep response concise, complete, beginner-friendly.
10. Always include a logical next step.
11. Never show technical details, email templates, or ticket drafts.
12. Never mention tickets unless instructed above.

If mood is NEGATIVE → be extra empathetic.
If mood is POSITIVE → match their energy.

Respond clearly. DO NOT list the 12 points."""

        answer = safe_generate(engineered_prompt, session_id=session)
        answer = clean_answer(answer)

        background_tasks.add_task(bg_save_message, session, "nor", answer)

        result = {
            "answer": answer,
            "needs_support": needs_support and not already_sent
        }

        if not needs_support and not ticket_pending.get(session):
            cache_set(ck, result)

        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "answer": "System temporarily under heavy load. Please try again shortly.",
            "needs_support": False
        }

@app.post("/api/ticket")
async def submit_ticket(request: TicketRequest, background_tasks: BackgroundTasks):
    try:
        history = get_history_as_text(request.session_id) or ""
        ticket_id = create_ticket(
            question=f"Name: {request.name}\nEmail: {request.email}\nSubject: {request.subject}\nDescription: {request.description}",
            answer=f"Conversation context:\n{history}",
            session_id=request.session_id
        )
        _store_ticket_record(request.session_id, request.subject)
        return {"ticket_id": ticket_id}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/feedback")
async def save_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    history = get_history_as_text(request.session_id) or ""
    background_tasks.add_task(
        bg_store_feedback,
        f"Session feedback: {history[:200]}",
        "feedback",
        request.rating / 5.0,
        {},
        request.session_id,
        request.rating,
        request.feedback
    )
    return {"status": "queued"}

@app.get("/api/status")
async def get_status():
    try:
        return get_usage_status()
    except Exception as e:
        return {"status": "error", "detail": str(e)}