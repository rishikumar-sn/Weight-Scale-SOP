from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any


class CaptureRepository:
    def __init__(self, database_path: Path, sessions_dir: Path) -> None:
        self.database_path = database_path
        self.sessions_dir = sessions_dir
        self._lock = RLock()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS captures (
                    id TEXT PRIMARY KEY,
                    captured_at TEXT NOT NULL,
                    weight_g REAL,
                    label TEXT,
                    status TEXT NOT NULL,
                    session_path TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def session_dir(self, capture_id: str) -> Path:
        return self.sessions_dir / capture_id

    def manifest_path(self, capture_id: str) -> Path:
        return self.session_dir(capture_id) / "manifest.json"

    def save(self, state: dict[str, Any]) -> dict[str, Any]:
        capture_id = str(state["id"])
        session_dir = self.session_dir(capture_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.manifest_path(capture_id)
        temporary = manifest.with_suffix(".tmp")
        with self._lock:
            temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
            temporary.replace(manifest)
            classification = state.get("classification") or {}
            label = classification.get("confirmed_label") or classification.get("predicted_label")
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO captures (id, captured_at, weight_g, label, status, session_path, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        weight_g=excluded.weight_g,
                        label=excluded.label,
                        status=excluded.status,
                        updated_at=excluded.updated_at
                    """,
                    (
                        capture_id,
                        state["captured_at"],
                        state.get("weight_g"),
                        label,
                        state.get("status", "captured"),
                        str(session_dir),
                        state.get("updated_at", state["captured_at"]),
                    ),
                )
        return deepcopy(state)

    def get(self, capture_id: str) -> dict[str, Any] | None:
        path = self.manifest_path(capture_id)
        if not path.is_file():
            return None
        with self._lock:
            return json.loads(path.read_text(encoding="utf-8"))

    def latest(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM captures ORDER BY captured_at DESC LIMIT 1"
            ).fetchone()
        return self.get(str(row["id"])) if row else None

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM captures ORDER BY captured_at DESC LIMIT ?",
                (max(1, min(100, limit)),),
            ).fetchall()
        return [dict(row) for row in rows]

