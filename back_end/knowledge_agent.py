import sys
import os
import sqlite3
import threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── GLOBAL MODEL (loaded once at startup) ──
_sentence_model = None
_model_lock = threading.Lock()

def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        with _model_lock:
            if _sentence_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
                    print("[Knowledge] SentenceTransformer loaded.")
                except Exception as e:
                    print(f"[Knowledge] Model load failed: {e}")
    return _sentence_model

# ── SAFE DB CONNECTION ──
_db_lock = threading.Lock()

def get_feedback_db():
    return sqlite3.connect("student_feedback.db", check_same_thread=False)

def search_knowledge(question: str) -> str:
    try:
        import chromadb
        model = get_sentence_model()
        if not model:
            return ""
        client = chromadb.PersistentClient(path="./knowledge_db")
        collection = client.get_or_create_collection(name="emsi_knowledge")
        if collection.count() == 0:
            documents = [
                "To request an attestation go to the administration office on the 2nd floor.",
                "Exam schedules are posted on the student portal 2 weeks before exams.",
                "To change your schedule contact your class delegate.",
                "The library is open from 8am to 8pm Monday to Friday.",
                "WiFi password can be obtained from the IT department.",
                "Internship documents must be submitted 2 weeks before the start date.",
                "To contest a grade submit a request within 72 hours after results.",
                "Absence justification must be submitted within 48 hours to administration.",
                "The student portal is accessible at portal.emsi.ma with your student ID.",
                "For administrative issues contact the secretary on the ground floor.",
                "Club activities are managed by the student council.",
                "End of year projects must be submitted before the deadline on the portal.",
            ]
            embeddings = model.encode(documents).tolist()
            ids = [f"doc_{i}" for i in range(len(documents))]
            collection.add(documents=documents, embeddings=embeddings, ids=ids)
        query_embedding = model.encode([question]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=2)
        if results and results["documents"] and results["documents"][0]:
            return " | ".join(results["documents"][0])
        return ""
    except Exception as e:
        print(f"[Knowledge] Chroma failed: {e}")
        return ""

# Common words to ignore in matching
_STOPWORDS = {
    "i","a","the","is","my","me","to","and","or","in","of","it","do","can",
    "je","le","la","les","un","une","des","et","est","pas","que","mon","ma",
    "هل","في","من","على","لا","هذا","أنا","مع"
}

def search_similar_feedback(question: str, min_rating: int = 1) -> str:
    try:
        q_words = set(question.lower().split()) - _STOPWORDS
        if not q_words:
            return ""
        with _db_lock:
            conn = get_feedback_db()
            cur = conn.cursor()
            cur.execute("""
                SELECT input_text, written_feedback, star_rating FROM feedback
                WHERE written_feedback IS NOT NULL
                ORDER BY star_rating DESC
                LIMIT 20
            """)
            rows = cur.fetchall()
            conn.close()

        if not rows:
            return ""

        scored = []
        for input_text, written_feedback, stars in rows:
            if not input_text:
                continue
            fb_words = set(input_text.lower().split()) - _STOPWORDS
            overlap = len(q_words & fb_words)
            if overlap > 0:
                scored.append((overlap, stars, written_feedback))

        scored.sort(key=lambda x: (-x[0], -x[1]))
        top = scored[:3]
        if not top:
            return ""
        return " | ".join(f"[{s}★] {fb}" for _, s, fb in top if fb)
    except Exception as e:
        print(f"[Feedback search] failed: {e}")
        return ""

def check_repeated_bad_feedback(session_id: str) -> bool:
    try:
        with _db_lock:
            conn = get_feedback_db()
            cur = conn.cursor()
            cur.execute("""
                SELECT COUNT(*) FROM feedback
                WHERE student_id = ? AND star_rating <= 2
            """, (session_id,))
            count = cur.fetchone()[0]
            conn.close()
        return count >= 3
    except Exception as e:
        print(f"[Bad feedback check] failed: {e}")
        return False