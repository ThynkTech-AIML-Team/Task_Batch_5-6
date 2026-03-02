import sqlite3

# Default DB name (can be overridden)
DB_NAME = "interview.db"

# 1️⃣ Initialize DB
def init_db(db_name=DB_NAME):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_name TEXT,
        candidate_id TEXT,
        question TEXT,
        answer TEXT,
        sentiment REAL,
        emotion TEXT,
        confidence INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

# 2️⃣ Save candidate response
def save_response(candidate_name, candidate_id, question, answer, sentiment_score, emotion_label, confidence_score, db_name=DB_NAME):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO responses (candidate_name, candidate_id, question, answer, sentiment, emotion, confidence)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (candidate_name, candidate_id, question, answer, sentiment_score, emotion_label, confidence_score))
    conn.commit()
    conn.close()

# 3️⃣ Fetch responses (all or specific candidate)
def fetch_responses(candidate_id=None, db_name=DB_NAME):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    if candidate_id:
        cursor.execute("SELECT * FROM responses WHERE candidate_id=?", (candidate_id,))
    else:
        cursor.execute("SELECT * FROM responses")
    data = cursor.fetchall()
    conn.close()
    return data