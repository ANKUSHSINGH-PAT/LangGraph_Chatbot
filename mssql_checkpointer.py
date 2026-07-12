"""
SQL Server checkpointer for LangGraph (synchronous).

Mirrors SqliteSaver's interface exactly but targets SQL Server via pyodbc.
Tables created automatically on first use:
  - checkpoints
  - writes
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Sequence, Tuple, cast

import pyodbc

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
    WRITES_IDX_MAP,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.utils import search_where
from langchain_core.runnables import RunnableConfig

import json
import random


# Connection string — Windows Auth to localhost
# _CONN_STR = (
#     "DRIVER={ODBC Driver 17 for SQL Server};"
#     "SERVER=localhost;"
#     "DATABASE=chatbot_db;"
#     "Trusted_Connection=yes;"
# )

_CONN_STR = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=LAPTOP-8LIOPLF8;"
    "DATABASE=chatbot_db;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)
class MSSQLSaver(BaseCheckpointSaver[str]):
    """LangGraph checkpointer backed by SQL Server (pyodbc, synchronous)."""

    conn: pyodbc.Connection
    is_setup: bool

    def __init__(self, conn: pyodbc.Connection, *, serde=None) -> None:
        super().__init__(serde=serde)
        self.jsonplus_serde = JsonPlusSerializer()
        self.conn = conn
        self.conn.autocommit = False
        self.is_setup = False
        self.lock = threading.Lock()

    @classmethod
    def from_conn_string(cls, conn_str: str = _CONN_STR) -> "MSSQLSaver":
        conn = pyodbc.connect(conn_str)
        return cls(conn)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def setup(self) -> None:
        if self.is_setup:
            return
        cur = self.conn.cursor()
        try:
            cur.execute(
                "IF OBJECT_ID('dbo.checkpoints', 'U') IS NULL "
                "CREATE TABLE dbo.checkpoints ("
                "thread_id NVARCHAR(255) NOT NULL,"
                "checkpoint_ns NVARCHAR(255) NOT NULL CONSTRAINT DF_cp_ns DEFAULT '',"
                "checkpoint_id NVARCHAR(255) NOT NULL,"
                "parent_checkpoint_id NVARCHAR(255) NULL,"
                "[type] NVARCHAR(255) NULL,"
                "[checkpoint] VARBINARY(MAX) NULL,"
                "metadata VARBINARY(MAX) NULL,"
                "PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)"
                ")"
            )
            cur.execute(
                "IF OBJECT_ID('dbo.writes', 'U') IS NULL "
                "CREATE TABLE dbo.writes ("
                "thread_id NVARCHAR(255) NOT NULL,"
                "checkpoint_ns NVARCHAR(255) NOT NULL CONSTRAINT DF_wr_ns DEFAULT '',"
                "checkpoint_id NVARCHAR(255) NOT NULL,"
                "task_id NVARCHAR(255) NOT NULL,"
                "idx INT NOT NULL,"
                "channel NVARCHAR(255) NOT NULL,"
                "[type] NVARCHAR(255) NULL,"
                "value VARBINARY(MAX) NULL,"
                "PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)"
                ")"
            )
            self.conn.commit()
        finally:
            cur.close()
        self.is_setup = True

    @contextmanager
    def cursor(self, transaction: bool = True) -> Iterator[pyodbc.Cursor]:
        with self.lock:
            self.setup()
            cur = self.conn.cursor()
            try:
                yield cur
                if transaction:
                    self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
            finally:
                cur.close()

    # ------------------------------------------------------------------
    # get_tuple
    # ------------------------------------------------------------------
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        with self.cursor(transaction=False) as cur:
            if checkpoint_id := get_checkpoint_id(config):
                cur.execute(
                    "SELECT thread_id, checkpoint_id, parent_checkpoint_id, [type], [checkpoint], metadata "
                    "FROM dbo.checkpoints WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ?",
                    (str(config["configurable"]["thread_id"]), checkpoint_ns, checkpoint_id),
                )
            else:
                cur.execute(
                    "SELECT TOP 1 thread_id, checkpoint_id, parent_checkpoint_id, [type], [checkpoint], metadata "
                    "FROM dbo.checkpoints WHERE thread_id = ? AND checkpoint_ns = ? "
                    "ORDER BY checkpoint_id DESC",
                    (str(config["configurable"]["thread_id"]), checkpoint_ns),
                )
            row = cur.fetchone()
            if not row:
                return None
            thread_id, checkpoint_id, parent_checkpoint_id, type_, checkpoint, metadata = row
            if not get_checkpoint_id(config):
                config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                }
            cur.execute(
                "SELECT task_id, channel, [type], value FROM dbo.writes "
                "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? "
                "ORDER BY task_id, idx",
                (
                    str(config["configurable"]["thread_id"]),
                    checkpoint_ns,
                    str(config["configurable"]["checkpoint_id"]),
                ),
            )
            pending_writes = [
                (tid, ch, self.serde.loads_typed((t, v)))
                for tid, ch, t, v in cur.fetchall()
            ]
            return CheckpointTuple(
                config,
                self.serde.loads_typed((type_, checkpoint)),
                cast(CheckpointMetadata, json.loads(metadata) if metadata else {}),
                (
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_checkpoint_id,
                        }
                    }
                    if parent_checkpoint_id
                    else None
                ),
                pending_writes,
            )

    # ------------------------------------------------------------------
    # list
    # ------------------------------------------------------------------
    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        where, param_values = search_where(config, filter, before)
        query = (
            "SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, [type], [checkpoint], metadata "
            f"FROM dbo.checkpoints {where} ORDER BY checkpoint_id DESC"
        )
        if limit:
            # SQL Server uses TOP, not LIMIT — inject after SELECT
            query = query.replace("SELECT ", f"SELECT TOP {limit} ", 1)
        with self.cursor(transaction=False) as cur:
            cur.execute(query, param_values)
            rows = cur.fetchall()
        for thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type_, checkpoint, metadata in rows:
            with self.cursor(transaction=False) as wcur:
                wcur.execute(
                    "SELECT task_id, channel, [type], value FROM dbo.writes "
                    "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? "
                    "ORDER BY task_id, idx",
                    (thread_id, checkpoint_ns, checkpoint_id),
                )
                pending = [(t, ch, self.serde.loads_typed((tp, v))) for t, ch, tp, v in wcur.fetchall()]
            yield CheckpointTuple(
                {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": checkpoint_id}},
                self.serde.loads_typed((type_, checkpoint)),
                cast(CheckpointMetadata, json.loads(metadata) if metadata else {}),
                (
                    {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": parent_checkpoint_id}}
                    if parent_checkpoint_id else None
                ),
                pending,
            )

    # ------------------------------------------------------------------
    # put
    # ------------------------------------------------------------------
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = json.dumps(get_checkpoint_metadata(config, metadata)).encode("utf-8")
        with self.cursor() as cur:
            cur.execute(
                "MERGE dbo.checkpoints AS target "
                "USING (SELECT ? AS thread_id, ? AS checkpoint_ns, ? AS checkpoint_id) AS src "
                "ON target.thread_id=src.thread_id AND target.checkpoint_ns=src.checkpoint_ns AND target.checkpoint_id=src.checkpoint_id "
                "WHEN MATCHED THEN "
                "  UPDATE SET parent_checkpoint_id=?, [type]=?, [checkpoint]=?, metadata=? "
                "WHEN NOT MATCHED THEN "
                "  INSERT (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, [type], [checkpoint], metadata) "
                "  VALUES (?, ?, ?, ?, ?, ?, ?);",
                (
                    str(thread_id), checkpoint_ns, checkpoint["id"],
                    config["configurable"].get("checkpoint_id"), type_, serialized_checkpoint, serialized_metadata,
                    str(thread_id), checkpoint_ns, checkpoint["id"],
                    config["configurable"].get("checkpoint_id"), type_, serialized_checkpoint, serialized_metadata,
                ),
            )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    # ------------------------------------------------------------------
    # put_writes
    # ------------------------------------------------------------------
    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        use_merge = all(w[0] in WRITES_IDX_MAP for w in writes)
        with self.cursor() as cur:
            for idx, (channel, value) in enumerate(writes):
                type_, serialized = self.serde.dumps_typed(value)
                real_idx = WRITES_IDX_MAP.get(channel, idx)
                if use_merge:
                    cur.execute(
                        "MERGE dbo.writes AS target "
                        "USING (SELECT ? AS thread_id, ? AS checkpoint_ns, ? AS checkpoint_id, ? AS task_id, ? AS idx) AS src "
                        "ON target.thread_id=src.thread_id AND target.checkpoint_ns=src.checkpoint_ns "
                        "   AND target.checkpoint_id=src.checkpoint_id AND target.task_id=src.task_id AND target.idx=src.idx "
                        "WHEN MATCHED THEN UPDATE SET channel=?, [type]=?, value=? "
                        "WHEN NOT MATCHED THEN INSERT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, [type], value) "
                        "    VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                        (
                            str(config["configurable"]["thread_id"]),
                            str(config["configurable"]["checkpoint_ns"]),
                            str(config["configurable"]["checkpoint_id"]),
                            task_id, real_idx,
                            channel, type_, serialized,
                            str(config["configurable"]["thread_id"]),
                            str(config["configurable"]["checkpoint_ns"]),
                            str(config["configurable"]["checkpoint_id"]),
                            task_id, real_idx, channel, type_, serialized,
                        ),
                    )
                else:
                    # INSERT IF NOT EXISTS
                    cur.execute(
                        "IF NOT EXISTS (SELECT 1 FROM dbo.writes WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=? AND task_id=? AND idx=?) "
                        "INSERT INTO dbo.writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, [type], value) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            str(config["configurable"]["thread_id"]),
                            str(config["configurable"]["checkpoint_ns"]),
                            str(config["configurable"]["checkpoint_id"]),
                            task_id, real_idx,
                            str(config["configurable"]["thread_id"]),
                            str(config["configurable"]["checkpoint_ns"]),
                            str(config["configurable"]["checkpoint_id"]),
                            task_id, real_idx, channel, type_, serialized,
                        ),
                    )

    # ------------------------------------------------------------------
    # version
    # ------------------------------------------------------------------
    def get_next_version(self, current: Optional[str], channel=None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    # ------------------------------------------------------------------
    # delete_thread_data
    # ------------------------------------------------------------------
    def delete_thread_data(self, thread_id: str) -> None:
        """Delete all checkpoints and writes for a given thread_id."""
        with self.cursor() as cur:
            cur.execute(
                "DELETE FROM dbo.writes WHERE thread_id = ?",
                (str(thread_id),),
            )
            cur.execute(
                "DELETE FROM dbo.checkpoints WHERE thread_id = ?",
                (str(thread_id),),
            )

    # ------------------------------------------------------------------
    # async stubs (not supported)
    # ------------------------------------------------------------------
    async def aget_tuple(self, config: RunnableConfig):
        raise NotImplementedError("Use get_tuple() — async not supported by MSSQLSaver.")

    async def alist(self, config, *, filter=None, before=None, limit=None) -> AsyncIterator:
        raise NotImplementedError("Use list() — async not supported by MSSQLSaver.")
        yield  # make it an async generator

    async def aput(self, config, checkpoint, metadata, new_versions):
        raise NotImplementedError("Use put() — async not supported by MSSQLSaver.")
