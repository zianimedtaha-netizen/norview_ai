import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv

load_dotenv()

TICKET_KEYWORDS = {
    "en": ["ticket","support","human","agent","complaint","not solved","still not working","help me","issue not fixed"],
    "fr": ["ticket","support","agent","humain","plainte","problème","pas résolu","ça marche pas","aidez-moi"],
    "ar": ["تذكرة","دعم","مشكلة","لم يتم الحل","مازال","مساعدة","بغيت مساعدة"],
    "dz": ["mouchkil","mochkila","ma t7alch","mazal","3tini chi wahed","bghit support","3afak 3awnni"],
}

def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())

def detect_ticket_keywords(text: str) -> bool:
    n = normalize(text)
    for lang_keywords in TICKET_KEYWORDS.values():
        for kw in lang_keywords:
            if kw in n:
                return True
    return False

def route_task(message: str, session_id: str = "default") -> dict:
    agent_result = ""
    knowledge = ""
    needs_support = False
    similar_feedback = ""

    # Sentiment
    try:
        from sentiments_agent import analyze_reviews
        from memory_agent import get_history_as_text
        history = get_history_as_text(session_id) or ""
        full_context = f"{history}\nLatest message: {message}"
        sentiment = analyze_reviews(full_context)
        if isinstance(sentiment, dict):
            label = sentiment.get("label", "NEUTRAL")
            confidence = sentiment.get("confidence", 0.5)
            agent_result = f"{label} ({confidence:.0%})"
            if label == "NEGATIVE" and confidence >= 0.75:
                needs_support = True
        else:
            agent_result = str(sentiment)
    except Exception as e:
        print(f"[Coordinator] Sentiment failed: {e}")

    # HIGHEST PRIORITY: multilingual keyword detection
    if detect_ticket_keywords(message):
        needs_support = True
    else:
        # Knowledge + similar feedback
        try:
            from knowledge_agent import search_knowledge, search_similar_feedback, check_repeated_bad_feedback
            emsi_keywords = [
                "attestation","schedule","library","wifi","internship","grade",
                "absence","portal","administration","club","project","exam",
                "secretary","policy","document"
            ]
            if any(word in message.lower() for word in emsi_keywords):
                knowledge = search_knowledge(message) or ""
            similar_feedback = search_similar_feedback(message) or ""
            if check_repeated_bad_feedback(session_id):
                needs_support = True
            # Low-rated similar feedback → increase escalation chance
            if not needs_support and similar_feedback:
                low_count = similar_feedback.count("1★") + similar_feedback.count("2★")
                if low_count >= 2:
                    needs_support = True
        except Exception as e:
            print(f"[Coordinator] Knowledge/feedback failed: {e}")

        # Gemini support check — ONLY if still False
        if not needs_support:
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                model = genai.GenerativeModel("gemini-2.5-flash")
                check = model.generate_content(
                    f'Does this student message indicate a serious problem needing EMSI staff help? Answer ONLY yes or no.\nMessage: "{message}"'
                )
                if check and hasattr(check, "text") and check.text:
                    needs_support = "yes" in check.text.lower().strip()
            except Exception as e:
                print(f"[Coordinator] Support check failed: {e}")

    if similar_feedback:
        agent_result += f" | SIMILAR_FEEDBACK: {similar_feedback}"

    return {
        "agent_result": agent_result,
        "knowledge": knowledge,
        "needs_support": needs_support
    }