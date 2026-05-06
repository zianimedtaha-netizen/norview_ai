import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def verify_response(question: str, answer: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    prompt = f"""
You are a fact-checking agent. A student asked a question and got an answer.
Your job is to verify if the answer is accurate and not hallucinated.

STUDENT QUESTION: {question}
ANSWER GIVEN: {answer}

Check if the answer:
1. Contains made up facts or statistics
2. Makes specific claims that seem uncertain
3. Contradicts common knowledge

If the answer is fine, return it exactly as is.
If it contains hallucinations, correct them and return the fixed answer.
Return ONLY the final answer, nothing else.
"""
    
    response = model.generate_content(prompt)
    return response.text