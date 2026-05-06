import numpy as np
import os
import json

# ── WEIGHT PERSISTENCE ──
WEIGHTS_FILE = "sentiment_weights.json"

# ── STUDENT-FOCUSED MULTILINGUAL TRAINING DATA ──
TRAINING_DATA = [
    # POSITIVE (1)
    ("I passed my exam I am so happy", 1),
    ("finally understood the lesson it feels amazing", 1),
    ("got a great grade on my project", 1),
    ("the teacher explained everything clearly merci", 1),
    ("passed all my tests this week feeling great", 1),
    ("j'ai réussi mon examen je suis trop content", 1),
    ("l'enseignant était super et le cours était intéressant", 1),
    ("نجحت في الامتحان الحمد لله", 1),
    ("كل شي مزيان هاد السيميستر", 1),
    ("wach njaht f exam iyeh 3afak", 1),
    ("motivated and ready to work hard this semester", 1),
    ("I love this class the teacher is wonderful", 1),

    # NEGATIVE (0)
    ("I failed my exam and feel terrible", 0),
    ("nobody listens to me in my group", 0),
    ("I studied hard but still got a bad grade", 0),
    ("the grading is completely unfair", 0),
    ("I have three deadlines and no time to sleep", 0),
    ("I do not understand anything from the lecture", 0),
    ("j'ai raté mon examen c'est catastrophique", 0),
    ("personne ne m'aide dans mon groupe c'est injuste", 0),
    ("je suis complètement perdu dans ce cours", 0),
    ("رسبت في الامتحان وما فهمتش والو", 0),
    ("الأستاذ ما كيشرحش مزيان وأنا مفهمتش", 0),
    ("mochkila kbira f blasti ma3arfch", 0),
    ("terrible experience I hate this situation", 0),
    ("I am so stressed I cannot handle the pressure", 0),
]

# ── KEYWORD FALLBACK ──
POSITIVE_WORDS = {
    "good","great","happy","excellent","passed","love","wonderful",
    "amazing","fantastic","merci","bien","super","réussi","نجحت",
    "مزيان","الحمد","iyeh","bravo","perfect","motivated"
}
NEGATIVE_WORDS = {
    "bad","terrible","failed","hate","awful","horrible","stressed",
    "lost","unfair","raté","perdu","injuste","رسبت","مفهمتش","mochkila",
    "problem","issue","help","crash","error","boring","difficult","hard"
}

def keyword_sentiment(text: str) -> dict:
    words = set(text.lower().split())
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    if pos > neg:
        return {"label": "POSITIVE", "confidence": 0.65, "score": 0.7}
    elif neg > pos:
        return {"label": "NEGATIVE", "confidence": 0.65, "score": 0.3}
    return {"label": "NEUTRAL", "confidence": 0.5, "score": 0.5}

# ── TF-IDF ──
def build_vocab(docs):
    vocab = set()
    for doc in docs:
        for word in doc.lower().split():
            vocab.add(word)
    return sorted(list(vocab))

def compute_tfidf(docs, vocab):
    n_docs = len(docs)
    n_vocab = len(vocab)
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    df = np.zeros(n_vocab)
    for doc in docs:
        for word in set(doc.lower().split()):
            if word in word_to_idx:
                df[word_to_idx[word]] += 1
    idf = np.log(n_docs / (df + 1))
    tfidf = np.zeros((n_docs, n_vocab))
    for i, doc in enumerate(docs):
        words = doc.lower().split()
        for word in words:
            if word in word_to_idx:
                j = word_to_idx[word]
                tf = words.count(word) / len(words)
                tfidf[i, j] = tf * idf[j]
    return tfidf, idf, word_to_idx

# ── NEURAL NETWORK ──
def relu(z): return np.maximum(0, z)
def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return a2, (z1, a1, z2, a2)

def compute_loss(y_pred, y_true):
    eps = 1e-8
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

def backward(X, y, cache, W2):
    z1, a1, z2, a2 = cache
    m = X.shape[0]
    dz2 = a2 - y
    dW2 = (a1.T @ dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m
    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)
    dW1 = (X.T @ dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2

# ── SAVE / LOAD WEIGHTS ──
def save_weights(W1, b1, W2, b2, vocab, idf, word_to_idx):
    data = {
        "W1": W1.tolist(), "b1": b1.tolist(),
        "W2": W2.tolist(), "b2": b2.tolist(),
        "vocab": vocab,
        "idf": idf.tolist(),
        "word_to_idx": word_to_idx
    }
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(data, f)
    print("[Sentiment] Weights saved.")

def load_weights():
    if not os.path.exists(WEIGHTS_FILE):
        return None
    try:
        with open(WEIGHTS_FILE, "r") as f:
            data = json.load(f)
        W1 = np.array(data["W1"])
        b1 = np.array(data["b1"])
        W2 = np.array(data["W2"])
        b2 = np.array(data["b2"])
        vocab = data["vocab"]
        idf = np.array(data["idf"])
        word_to_idx = data["word_to_idx"]
        print("[Sentiment] Weights loaded from file.")
        return W1, b1, W2, b2, vocab, idf, word_to_idx
    except Exception as e:
        print(f"[Sentiment] Load failed: {e}")
        return None

def train():
    texts = [d[0] for d in TRAINING_DATA]
    labels = [d[1] for d in TRAINING_DATA]
    vocab = build_vocab(texts)
    X, idf, word_to_idx = compute_tfidf(texts, vocab)
    y = np.array(labels).reshape(-1, 1)
    np.random.seed(42)
    n_input = X.shape[1]
    n_hidden = 16
    W1 = np.random.randn(n_input, n_hidden) * np.sqrt(2.0 / n_input)
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, 1) * np.sqrt(1.0 / n_hidden)
    b2 = np.zeros((1, 1))
    for _ in range(300):
        pred, cache = forward(X, W1, b1, W2, b2)
        dW1, db1_g, dW2, db2_g = backward(X, y, cache, W2)
        W1 -= 0.1 * dW1; b1 -= 0.1 * db1_g
        W2 -= 0.1 * dW2; b2 -= 0.1 * db2_g
    save_weights(W1, b1, W2, b2, vocab, idf, word_to_idx)
    print("[Sentiment] Training complete.")
    return W1, b1, W2, b2, vocab, idf, word_to_idx
# ── INIT DB ──
def init_db():
    try:
        import sqlite3
        conn = sqlite3.connect("student_feedback.db")
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                input_text TEXT,
                detected_emotion TEXT,
                confidence REAL,
                all_scores TEXT,
                star_rating INTEGER,
                written_feedback TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
        conn.close()
        print("[DB] Feedback table ready.")
    except Exception as e:
        print(f"[DB] Init failed: {e}")

init_db()

# ── STARTUP: load or train ──
loaded = load_weights()
if loaded:
    W1, b1, W2, b2, vocab, idf, word_to_idx = loaded
else:
    W1, b1, W2, b2, vocab, idf, word_to_idx = train()

def analyze_reviews(text: str) -> dict:
    """Returns {label, confidence, score}"""
    try:
        words = text.lower().split()
        n_vocab = len(vocab)
        vec = np.zeros((1, n_vocab))
        for word in words:
            if word in word_to_idx:
                j = word_to_idx[word]
                tf = words.count(word) / len(words)
                vec[0, j] = tf * idf[j]
        pred, _ = forward(vec, W1, b1, W2, b2)
        score = float(pred[0, 0])
        if score > 0.6:
            return {"label": "POSITIVE", "confidence": round(score, 2), "score": score}
        elif score < 0.4:
            return {"label": "NEGATIVE", "confidence": round(1 - score, 2), "score": score}
        else:
            return {"label": "NEUTRAL", "confidence": round(1 - abs(score - 0.5) * 2, 2), "score": score}
    except Exception as e:
        print(f"[Sentiment] NN failed, using keyword fallback: {e}")
        return keyword_sentiment(text)

def store_feedback(input_text, detected_emotion, confidence, all_scores,
                   student_id=None, star_rating=None, written_feedback=None):
    """Saves feedback to SQLite — called from main.py /api/feedback"""
    try:
        import sqlite3
        conn = sqlite3.connect("student_feedback.db")
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                input_text TEXT,
                detected_emotion TEXT,
                confidence REAL,
                all_scores TEXT,
                star_rating INTEGER,
                written_feedback TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        cur.execute("""
            INSERT INTO feedback (student_id, input_text, detected_emotion,
                confidence, all_scores, star_rating, written_feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (student_id, input_text, detected_emotion,
              confidence, str(all_scores), star_rating, written_feedback))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Feedback DB] store failed: {e}")