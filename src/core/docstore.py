import sqlite3
import os
from typing import Optional, Dict, List


class LocalDocStore:
    """
    Lightweight SQLite-based Document Store for parent sections.
    Enables Small-to-Big retrieval without vector DB metadata bloat.
    """

    def __init__(self, db_path: str = "./data/docstore.sqlite"):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS parents (parent_id TEXT PRIMARY KEY, parent_text TEXT)"
        )
        self.conn.commit()

    def set(self, parent_id: str, parent_text: str):
        """Store or update parent text."""
        self.conn.execute(
            "REPLACE INTO parents (parent_id, parent_text) VALUES (?, ?)",
            (parent_id, parent_text),
        )
        self.conn.commit()

    def set_batch(self, parent_dict: Dict[str, str]):
        """Store multiple parent sections in a single transaction."""
        with self.conn:
            self.conn.executemany(
                "REPLACE INTO parents (parent_id, parent_text) VALUES (?, ?)",
                list(parent_dict.items()),
            )

    def get(self, parent_id: str) -> Optional[str]:
        """Retrieve parent text by parent_id."""
        cursor = self.conn.execute(
            "SELECT parent_text FROM parents WHERE parent_id = ?",
            (parent_id,),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def get_batch(self, parent_ids: List[str]) -> Dict[str, str]:
        """Retrieve multiple parent texts."""
        if not parent_ids:
            return {}
        placeholders = ",".join(["?"] * len(parent_ids))
        cursor = self.conn.execute(
            f"SELECT parent_id, parent_text FROM parents WHERE parent_id IN ({placeholders})",
            parent_ids,
        )
        return {row[0]: row[1] for row in cursor.fetchall()}

    def clear(self):
        """Clear all entries in the docstore."""
        with self.conn:
            self.conn.execute("DELETE FROM parents")

    def close(self):
        """Close SQLite connection."""
        self.conn.close()

