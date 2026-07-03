"""
Thread metadata store backed by SQL Server (pyodbc).
Stores thread_id, title, and created_at for the chatbot sidebar.
"""
import pyodbc
from datetime import datetime

_CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=chatbot_db;"
    "Trusted_Connection=yes;"
)


def _connect() -> pyodbc.Connection:
    return pyodbc.connect(_CONN_STR)


def init_db() -> None:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        IF OBJECT_ID('dbo.threads', 'U') IS NULL
        CREATE TABLE dbo.threads (
            thread_id  NVARCHAR(255) PRIMARY KEY,
            title      NVARCHAR(500),
            created_at NVARCHAR(50)
        )
    """)
    conn.commit()
    cur.close()
    conn.close()


def save_thread(thread_id: str, title: str) -> None:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        MERGE dbo.threads AS target
        USING (SELECT ? AS thread_id) AS src
        ON target.thread_id = src.thread_id
        WHEN MATCHED THEN
            UPDATE SET title = ?, created_at = ?
        WHEN NOT MATCHED THEN
            INSERT (thread_id, title, created_at)
            VALUES (?, ?, ?);
    """, (
        str(thread_id),
        title, datetime.utcnow().isoformat(),
        str(thread_id), title, datetime.utcnow().isoformat(),
    ))
    conn.commit()
    cur.close()
    conn.close()


def get_threads() -> list[tuple[str, str]]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT thread_id, title
        FROM dbo.threads
        ORDER BY created_at DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [(r[0], r[1]) for r in rows]
