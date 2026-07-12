"""
Thread metadata store backed by SQL Server (pyodbc).
Stores thread_id, title, and created_at for the chatbot sidebar.
"""
import pyodbc
import threading
from datetime import datetime

_CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=LAPTOP-8LIOPLF8;"
    "DATABASE=chatbot_db;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

# Connection pooling - reuse single connection
_conn_lock = threading.Lock()
_cached_conn: pyodbc.Connection | None = None


def _connect() -> pyodbc.Connection:
    """Get or create a reusable database connection."""
    global _cached_conn
    if _cached_conn is None:
        with _conn_lock:
            if _cached_conn is None or _cached_conn.closed:
                _cached_conn = pyodbc.connect(_CONN_STR, autocommit=False)
    return _cached_conn


def init_db() -> None:
    conn = _connect()
    cur = conn.cursor()
    try:
        cur.execute("""
            IF OBJECT_ID('dbo.threads', 'U') IS NULL
            CREATE TABLE dbo.threads (
                thread_id  NVARCHAR(255) PRIMARY KEY,
                title      NVARCHAR(500),
                created_at NVARCHAR(50)
            )
        """)
        conn.commit()
    finally:
        cur.close()


def save_thread(thread_id: str, title: str) -> None:
    conn = _connect()
    cur = conn.cursor()
    try:
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
    finally:
        cur.close()


def get_threads() -> list[tuple[str, str]]:
    conn = _connect()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT thread_id, title
            FROM dbo.threads
            ORDER BY created_at DESC
        """)
        rows = cur.fetchall()
        return [(r[0], r[1]) for r in rows]
    finally:
        cur.close()


def delete_thread(thread_id: str) -> None:
    """Delete a thread row from the threads table."""
    conn = _connect()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM dbo.threads WHERE thread_id = ?", (str(thread_id),))
        conn.commit()
    finally:
        cur.close()
