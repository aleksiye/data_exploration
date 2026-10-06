"""A small web dashboard for the follow-up team.

    python -m ztf_pipeline dashboard        then open http://127.0.0.1:8050

It shows each closed night's selected candidates as a worklist (to do / observed / skipped, with a note
and who did it), lists all earlier nights, and searches every classified object by ZTF ID. It reads
the classifier service's audit log on each request, so new nights appear on their own, and stores the
team's progress in output/worklist.db. Standard library only.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import Settings

log = logging.getLogger(__name__)

PAGE = Path(__file__).with_name("dashboard.html")
STATUSES = ("todo", "observed", "skipped")
MAX_NOTE, MAX_NAME, MAX_BODY = 500, 40, 10_000
SEARCH_LIMIT = 50


class BadRequest(ValueError):
    pass


class Worklist:
    """The team's progress on each selected candidate, kept in a SQLite file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS progress (
                    night      TEXT NOT NULL,
                    ztf_id     TEXT NOT NULL,
                    status     TEXT NOT NULL,
                    note       TEXT NOT NULL DEFAULT '',
                    updated_by TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (night, ztf_id)
                )""")

    def for_night(self, night: str) -> dict[str, dict]:
        with self._lock:
            rows = self._db.execute("SELECT ztf_id, status, note, updated_by, updated_at FROM progress "
                                    "WHERE night = ?", (night,)).fetchall()
        return {z: {"status": s, "note": n, "updated_by": b, "updated_at": t} for z, s, n, b, t in rows}

    def handled_per_night(self) -> dict[str, dict[str, int]]:
        with self._lock:
            rows = self._db.execute("SELECT night, status, COUNT(*) FROM progress WHERE status != 'todo' "
                                    "GROUP BY night, status").fetchall()
        out: dict[str, dict[str, int]] = {}
        for night, status, n in rows:
            out.setdefault(night, {})[status] = n
        return out

    def update(self, night: str, ztf_id: str, status: str, note: str, by: str) -> dict:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO progress (night, ztf_id, status, note, updated_by, updated_at) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (night, ztf_id) DO UPDATE SET status = excluded.status, note = excluded.note, "
                "updated_by = excluded.updated_by, updated_at = excluded.updated_at",
                (night, ztf_id, status, note, by, now))
        return {"status": status, "note": note, "updated_by": by, "updated_at": now}

    def close(self) -> None:
        self._db.close()


class AuditData:
    """The classifier service's audit log, re-read only when a file changes."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._cache: dict[Path, tuple[tuple[float, int], list[dict]]] = {}
        self._lock = threading.Lock()

    def _read(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        stat = path.stat()
        key = (stat.st_mtime, stat.st_size)
        with self._lock:
            cached = self._cache.get(path)
            if cached and cached[0] == key:
                return cached[1]
            with open(path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
            self._cache[path] = (key, records)
            return records

    def nights(self) -> dict[str, dict]:
        """Final selection per night (the latest one, as delivery is at-least-once)."""
        return {s["night"]: s for s in self._read(self.settings.selections_log) if s.get("final")}

    def classifications(self) -> dict[str, dict]:
        return {c["ztf_id"]: c for c in self._read(self.settings.classifications_log)}


class Dashboard:
    """The API behind the page. Every method returns JSON-ready data."""

    def __init__(self, settings: Settings):
        self.data = AuditData(settings)
        self.worklist = Worklist(settings.worklist_path)

    def list_nights(self) -> list[dict]:
        handled = self.worklist.handled_per_night()
        nights = []
        for night, s in sorted(self.data.nights().items(), reverse=True):
            done = handled.get(night, {})
            nights.append({"night": night, "selected": len(s["candidates"]), "alerts": s["alerts_tonight"],
                           "eligible": s["eligible_tonight"], "observed": done.get("observed", 0),
                           "skipped": done.get("skipped", 0)})
        return nights

    def night(self, night: str) -> dict:
        selection = self.data.nights().get(night)
        if selection is None:
            raise BadRequest(f"no closed night {night!r}")
        progress = self.worklist.for_night(night)
        todo = {"status": "todo", "note": "", "updated_by": "", "updated_at": ""}
        candidates = [{**c, **progress.get(c["ztf_id"], todo)} for c in selection["candidates"]]
        summary = {k: selection[k] for k in ("night", "model_version", "threshold", "capacity",
                                             "alerts_tonight", "eligible_tonight", "generated_at")}
        return {**summary, "candidates": candidates}

    def search(self, query: str) -> list[dict]:
        """Every classified object whose ZTF ID contains the query, with where it was selected (if at all)."""
        query = query.strip().upper()
        if len(query) < 3:
            raise BadRequest("type at least 3 characters")
        selected = {(c["ztf_id"]): (night, c["rank"]) for night, s in self.data.nights().items()
                    for c in s["candidates"]}
        matches = [c for z, c in self.data.classifications().items() if query in z.upper()]
        matches.sort(key=lambda c: (c["night"], c["p_ia"]), reverse=True)       # newest night, best first
        results = []
        for c in matches[:SEARCH_LIMIT]:
            night, rank = selected.get(c["ztf_id"], (None, None))
            status = self.worklist.for_night(night).get(c["ztf_id"], {}).get("status", "todo") if night else None
            results.append({"ztf_id": c["ztf_id"], "night": c["night"], "p_ia": c["p_ia"], "eligible": c["eligible"],
                            "selected_night": night, "rank": rank, "status": status})
        return results

    def update(self, body: dict) -> dict:
        night, ztf_id = str(body.get("night", "")), str(body.get("ztf_id", ""))
        status, note, by = body.get("status"), str(body.get("note", "")), str(body.get("by", "")).strip()
        if status not in STATUSES:
            raise BadRequest(f"status must be one of {STATUSES}")
        if len(note) > MAX_NOTE or len(by) > MAX_NAME:
            raise BadRequest("note or name too long")
        selection = self.data.nights().get(night)
        if selection is None or ztf_id not in {c["ztf_id"] for c in selection["candidates"]}:
            raise BadRequest(f"{ztf_id!r} was not selected on {night!r}")
        log.info("%s marked %s %s as %s", by or "someone", night, ztf_id, status)
        return self.worklist.update(night, ztf_id, status, note, by)


def make_handler(dashboard: Dashboard) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ztf-dashboard"

        def do_GET(self) -> None:
            url = urlparse(self.path)
            params = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/nights":
                    self._json(dashboard.list_nights())
                elif url.path == "/api/night":
                    self._json(dashboard.night(params.get("night", "")))
                elif url.path == "/api/search":
                    self._json(dashboard.search(params.get("q", "")))
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except BadRequest as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/progress":
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._json({"error": "request too large"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise BadRequest("expected a JSON object")
                self._json(dashboard.update(body))
            except (BadRequest, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def _json(self, payload, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            log.debug("%s - " + fmt, self.address_string(), *args)

    return Handler


def make_server(settings: Settings, host: str = "127.0.0.1", port: int = 8050) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(Dashboard(settings)))
