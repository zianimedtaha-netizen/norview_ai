# norview_ai

An AI-powered project built with FastAPI and Google Generative AI.

---

## Requirements

- Python 3.14+
- Git

---

## Installation

### 1. Clone the repository

**Windows & Linux:**

    git clone https://github.com/zianimedtaha-netizen/norview_ai.git
    cd norview_ai

### 2. Install dependencies

**Windows:**

    py -m pip install -r requirements.txt

**Linux:**

    python3 -m pip install -r requirements.txt

### 3. Set up environment variables

Create a `.env` file in the root folder and add your API keys:

    GOOGLE_API_KEY=your_api_key_here

### 4. Run the server

**Windows:**

    py -m uvicorn back_end.main:app --reload

**Linux:**

    python3 -m uvicorn back_end.main:app --reload

The server will start at: `http://127.0.0.1:8000`

### 5. Open the frontend

Open `front_end/index.html` in your browser with liveserver

---

## Project Structure

    norview_ai/
├── .venv/
├── back_end/
│   ├── __pycache__/
│   ├── knowledge_db/
│   ├── anti_hallucination_agent
│   ├── coordinator_agent.py
│   ├── knowledge_agent.py
│   ├── main.py
│   ├── memory_agent.py
│   ├── requirements.txt
│   ├── sentiment_weights.js
│   ├── sentiments_agent.py
│   ├── student_feedback.db
│   ├── ticket_agent.py
│   ├── tickets.db
│   └── token_guard_agent.py
├── front_end/
│   └── index.html
├── .env
├── .gitignore
├── README.md
└── requirements.txt 

---

## Tech Stack

- Python
- FastAPI + Uvicorn
- Google Generative AI
- Pydantic
- Python-dotenv
