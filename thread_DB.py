import sqlite3
from datetime import datetime

DB = "threads.db"

def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS threads (
        thread_id TEXT PRIMARY KEY,
        title TEXT,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()


def save_thread(thread_id, title):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
    INSERT OR REPLACE INTO threads VALUES (?, ?, ?)
    """, (thread_id, title, datetime.utcnow().isoformat()))

    conn.commit()
    conn.close()


def get_threads():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    cur.execute("""
    SELECT thread_id, title
    FROM threads
    ORDER BY created_at DESC
    """)

    rows = cur.fetchall()
    conn.close()
    return rows
