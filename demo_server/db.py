"""Demo 任务库。记每一次听写或翻译，以及每部影片的播放进度。"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "demo.sqlite"

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init() -> None:
    with _lock, connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              id            TEXT PRIMARY KEY,
              kind          TEXT NOT NULL,
              video_path    TEXT NOT NULL,
              stream_index  INTEGER NOT NULL,
              language      TEXT NOT NULL,
              target_lang   TEXT,
              client_type   TEXT,
              status        TEXT NOT NULL,
              stage         TEXT,
              done_count    INTEGER NOT NULL DEFAULT 0,
              total_count   INTEGER,
              error         TEXT,
              created_at    TEXT NOT NULL,
              updated_at    TEXT NOT NULL,
              finished_at   TEXT
            )
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        if "log" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN log TEXT NOT NULL DEFAULT ''")
        if "model" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN model TEXT")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS playback (
              video_path    TEXT PRIMARY KEY,
              position_sec  REAL NOT NULL,
              updated_at    TEXT NOT NULL
            )
            """
        )
        playback_columns = {row[1] for row in conn.execute("PRAGMA table_info(playback)")}
        if "subtitle_id" not in playback_columns:
            conn.execute("ALTER TABLE playback ADD COLUMN subtitle_id TEXT")


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def running_job() -> dict[str, Any] | None:
    with _lock, connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status = 'running' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    return row_to_dict(row)


def get_job(job_id: str) -> dict[str, Any] | None:
    with _lock, connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return row_to_dict(row)


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with _lock, connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [row_to_dict(row) for row in rows]  # type: ignore[misc]


def create_job(
    *,
    kind: str,
    video_path: str,
    stream_index: int,
    language: str,
    target_lang: str | None,
    client_type: str | None,
    model: str | None = None,
) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    now = _now()
    with _lock, connect() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
              id, kind, video_path, stream_index, language, target_lang,
              client_type, model, status, stage, done_count, total_count,
              created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, 0, NULL, ?, ?)
            """,
            (
                job_id,
                kind,
                video_path,
                stream_index,
                language,
                target_lang,
                client_type,
                model,
                "extract" if kind == "transcribe" else "translate",
                now,
                now,
            ),
        )
    job = get_job(job_id)
    assert job is not None
    return job


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = _now()
    columns = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values()) + [job_id]
    with _lock, connect() as conn:
        conn.execute(f"UPDATE jobs SET {columns} WHERE id = ?", values)


def append_log(job_id: str, message: str) -> None:
    stamp = datetime.now().astimezone().strftime("%H:%M:%S")
    line = f"{stamp}  {message}"
    now = _now()
    with _lock, connect() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET log = CASE
                  WHEN log IS NULL OR log = '' THEN ?
                  ELSE log || char(10) || ?
                END,
                updated_at = ?
            WHERE id = ?
            """,
            (line, line, now, job_id),
        )


def remove_job(job_id: str) -> None:
    with _lock, connect() as conn:
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


def playback_position(video_path: str) -> float:
    return float(playback_prefs(video_path)["positionSec"])


def playback_prefs(video_path: str) -> dict[str, Any]:
    with _lock, connect() as conn:
        row = conn.execute(
            "SELECT position_sec, subtitle_id FROM playback WHERE video_path = ?",
            (video_path,),
        ).fetchone()
    if row is None:
        return {"positionSec": 0.0, "subtitleId": ""}
    return {
        "positionSec": max(0.0, float(row["position_sec"] or 0)),
        "subtitleId": str(row["subtitle_id"] or ""),
    }


def save_playback(video_path: str, position_sec: float) -> None:
    now = _now()
    position = max(0.0, float(position_sec))
    with _lock, connect() as conn:
        conn.execute(
            """
            INSERT INTO playback (video_path, position_sec, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(video_path) DO UPDATE SET
              position_sec = excluded.position_sec,
              updated_at = excluded.updated_at
            """,
            (video_path, position, now),
        )


def save_subtitle(video_path: str, subtitle_id: str) -> None:
    now = _now()
    chosen = str(subtitle_id or "")
    with _lock, connect() as conn:
        conn.execute(
            """
            INSERT INTO playback (video_path, position_sec, subtitle_id, updated_at)
            VALUES (?, 0, ?, ?)
            ON CONFLICT(video_path) DO UPDATE SET
              subtitle_id = excluded.subtitle_id,
              updated_at = excluded.updated_at
            """,
            (video_path, chosen, now),
        )


def finish_job(job_id: str, status: str, error: str | None = None) -> None:
    now = _now()
    with _lock, connect() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = ?, error = ?, finished_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, error, now, now, job_id),
        )
