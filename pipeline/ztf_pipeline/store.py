"""Tonight's candidates, kept in an in-memory SQLite database."""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable

SCHEMA = """
CREATE TABLE candidates (
    ztf_id     TEXT PRIMARY KEY,
    night      TEXT NOT NULL,
    p_ia       REAL NOT NULL,
    event_time TEXT NOT NULL,
    details    TEXT NOT NULL    -- JSON: what the team needs to observe it
);
CREATE INDEX candidates_by_night_and_score ON candidates (night, p_ia DESC);
"""


class CandidateStore:
    def __init__(self):
        self._db = sqlite3.connect(":memory:")
        self._db.executescript(SCHEMA)

    def upsert(self, rows: Iterable[tuple[str, str, float, str, dict]]) -> None:
        """Insert (ztf_id, night, p_ia, event_time, details) rows. A repeated alert replaces the earlier one."""
        with self._db:
            self._db.executemany(
                "INSERT INTO candidates (ztf_id, night, p_ia, event_time, details) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (ztf_id) DO UPDATE SET night = excluded.night, p_ia = excluded.p_ia, "
                "event_time = excluded.event_time, details = excluded.details",
                ((z, n, p, t, json.dumps(d)) for z, n, p, t, d in rows),
            )

    def count(self, night: str, min_p: float = 0.0) -> int:
        (n,) = self._db.execute("SELECT COUNT(*) FROM candidates WHERE night = ? AND p_ia >= ?",
                                (night, min_p)).fetchone()
        return n

    def top(self, night: str, min_p: float, limit: int) -> list[tuple[str, float, dict]]:
        """The night's highest-scoring candidates at or above min_p. Ties go to the earlier alert."""
        rows = self._db.execute(
            "SELECT ztf_id, p_ia, details FROM candidates WHERE night = ? AND p_ia >= ? "
            "ORDER BY p_ia DESC, event_time ASC, ztf_id ASC LIMIT ?",
            (night, min_p, limit),
        ).fetchall()
        return [(z, p, json.loads(d)) for z, p, d in rows]

    def evict(self, night: str) -> int:
        with self._db:
            return self._db.execute("DELETE FROM candidates WHERE night = ?", (night,)).rowcount

    def close(self) -> None:
        self._db.close()
