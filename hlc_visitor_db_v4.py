"""
hlc_visitor_db_v4.py
─────────────────────
Visitor DB updated for multi-door HLC.

Key additions vs v2:
  - door_id column in visitor_sessions (which door was used to enter)
  - exit_door column (which door was used to exit — may differ!)
  - exit_method column: "CAMERA" | "TIMEOUT" | "GAP_REENTRY" | "MANUAL"
  - close_session_with_method() accepts exit_door parameter
  - get_door_stats() — per-door traffic analysis

All existing methods preserved.
"""

import sqlite3
import threading
import logging
from datetime import datetime, date
from typing import Optional, List, Dict
from config import config
from contextlib import contextmanager

logger = logging.getLogger(__name__)

VISITOR_DB_PATH = "hlc_visitors.db"


class HLCVisitorDBV4:
    def __init__(self, db_path: str = VISITOR_DB_PATH):
        self.db_path = db_path
        self._local  = threading.local()
        self._init()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            c = sqlite3.connect(self.db_path, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            self._local.conn = c
        return self._local.conn

    @contextmanager
    def _cur(self):
        conn = self._conn()
        cur  = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init(self):
        with self._cur() as c:

            c.execute("""
                CREATE TABLE IF NOT EXISTS visitor_sessions (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    visitor_id       TEXT NOT NULL,
                    name             TEXT,
                    dept             TEXT,
                    designation      TEXT DEFAULT '',
                    person_type      TEXT DEFAULT 'Teaching',

                    visit_date       TEXT NOT NULL,
                    entry_time       TEXT NOT NULL,
                    exit_time        TEXT,
                    duration_minutes REAL,

                    -- Door tracking (key for multi-door setup)
                    door_id          TEXT,          -- entry door: "DOOR_A" or "DOOR_B"
                    exit_door        TEXT,          -- exit door (may differ from entry door)
                    camera_id        TEXT,          -- entry camera
                    exit_camera_id   TEXT,          -- exit camera

                    -- Exit metadata
                    exit_method      TEXT DEFAULT 'CAMERA',
                                                   -- CAMERA | TIMEOUT | GAP_REENTRY | MANUAL

                    visit_number     INTEGER DEFAULT 1,
                    flagged          INTEGER DEFAULT 1,
                    over_15min       INTEGER DEFAULT 0,
                    purpose          TEXT,
                    confidence       REAL,

                    alert_sent       INTEGER DEFAULT 0,
                    alert_id         INTEGER,

                    created_at       TEXT DEFAULT (datetime('now','localtime'))
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_vs_vid  ON visitor_sessions(visitor_id, visit_date)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_vs_date ON visitor_sessions(visit_date)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_vs_open ON visitor_sessions(visit_date, exit_time)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_vs_door ON visitor_sessions(door_id, visit_date)")

            c.execute("""
                CREATE TABLE IF NOT EXISTS visit_alerts (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id      INTEGER NOT NULL,
                    visitor_id      TEXT NOT NULL,
                    visitor_name    TEXT,
                    dept            TEXT,
                    visit_date      TEXT NOT NULL,
                    entry_time      TEXT NOT NULL,
                    triggered_at    TEXT NOT NULL,
                    duration_at_trigger REAL,
                    status          TEXT DEFAULT 'OPEN',
                    resolved_at     TEXT,
                    resolved_by     TEXT,
                    resolution_note TEXT,
                    notified_dashboard INTEGER DEFAULT 0,
                    notified_whatsapp  INTEGER DEFAULT 0,
                    notified_email     INTEGER DEFAULT 0,
                    created_at      TEXT DEFAULT (datetime('now','localtime')),
                    FOREIGN KEY(session_id) REFERENCES visitor_sessions(id)
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_va_status ON visit_alerts(status)")

            # Migration: add designation column if this db pre-dates it
            # (existing hlc_visitors.db on the HP tower won't have this yet)
            existing_cols = {row[1] for row in c.execute("PRAGMA table_info(visitor_sessions)")}
            if "designation" not in existing_cols:
                c.execute("ALTER TABLE visitor_sessions ADD COLUMN designation TEXT DEFAULT ''")
                logger.info("Migrated visitor_sessions: added designation column")

            existing_va_cols = {row[1] for row in c.execute("PRAGMA table_info(visit_alerts)")}
            if "notified_email" not in existing_va_cols:
                c.execute("ALTER TABLE visit_alerts ADD COLUMN notified_email INTEGER DEFAULT 0")
                logger.info("Migrated visit_alerts: added notified_email column")

            c.execute("""
                CREATE TABLE IF NOT EXISTS alert_log (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id  INTEGER,
                    channel   TEXT,
                    recipient TEXT,
                    message   TEXT,
                    status    TEXT,
                    sent_at   TEXT DEFAULT (datetime('now','localtime'))
                )
            """)
        logger.info(f"HLC Visitor DB v4 initialised at {self.db_path}")

    # ── Session open ──────────────────────────────────────────────────────────

    def open_session(self, visitor_id: str, name: str, dept: str,
                      person_type: str, visit_date: str, entry_time: str,
                      designation: str = "",
                      confidence: float = 0.0,
                      camera_id: str = "",
                      door_id: str = "") -> int:
        with self._cur() as c:
            c.execute(
                "SELECT COUNT(*) as n FROM visitor_sessions "
                "WHERE visitor_id=? AND visit_date=?",
                (visitor_id, visit_date)
            )
            visit_number = c.fetchone()["n"] + 1

            c.execute("""
                INSERT INTO visitor_sessions
                    (visitor_id, name, dept, designation, person_type, visit_date,
                     entry_time, visit_number, confidence, camera_id, door_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (visitor_id, name, dept, designation, person_type, visit_date,
                  entry_time, visit_number, confidence, camera_id, door_id))
            sid = c.lastrowid
            logger.info(
                f"SESSION OPEN | {visitor_id} | door={door_id} "
                f"visit#{visit_number} | id={sid}"
            )
            return sid

    # ── Session close ─────────────────────────────────────────────────────────

    def close_session_with_method(self, session_id: int,
                                   exit_time: str,
                                   exit_method: str = "CAMERA",
                                   exit_door: str = "",
                                   exit_camera_id: str = "") -> Dict:
        with self._cur() as c:
            c.execute("SELECT * FROM visitor_sessions WHERE id=?",
                      (session_id,))
            row = c.fetchone()
            if not row:
                return {}

            entry = row["entry_time"]
            try:
                dur = (datetime.strptime(exit_time, "%H:%M:%S") -
                       datetime.strptime(entry,     "%H:%M:%S")
                       ).total_seconds() / 60.0
                dur = round(dur, 2)
            except Exception:
                dur = None

            # Field/column name "over_15min" is legacy — logic now uses the
            # actual configurable threshold (config.hlc.alert_threshold_mins),
            # not a hardcoded 15. Renaming the column would need a migration;
            # not worth it since only the behavior matters here.
            over_15 = 1 if (dur and dur > config.hlc.alert_threshold_mins) else 0

            c.execute("""
                UPDATE visitor_sessions
                SET exit_time=?, duration_minutes=?, over_15min=?,
                    exit_method=?, exit_door=?, exit_camera_id=?
                WHERE id=?
            """, (exit_time, dur, over_15,
                  exit_method, exit_door, exit_camera_id,
                  session_id))

            entry_door = row["door_id"] or "?"
            logger.info(
                f"SESSION CLOSE | id={session_id} | "
                f"entry_door={entry_door} exit_door={exit_door} | "
                f"method={exit_method} | dur={dur}m"
            )
            return {
                "session_id":   session_id,
                "visitor_id":   row["visitor_id"],
                "entry_door":   entry_door,
                "exit_door":    exit_door,
                "exit_time":    exit_time,
                "duration_min": dur,
                "over_15min":   bool(over_15),
                "exit_method":  exit_method,
            }

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_open_session(self, visitor_id: str,
                          visit_date: str) -> Optional[Dict]:
        with self._cur() as c:
            c.execute("""
                SELECT * FROM visitor_sessions
                WHERE visitor_id=? AND visit_date=? AND exit_time IS NULL
                ORDER BY id DESC LIMIT 1
            """, (visitor_id, visit_date))
            row = c.fetchone()
            return dict(row) if row else None

    def get_all_open_sessions(self, visit_date: str) -> List[Dict]:
        with self._cur() as c:
            c.execute("""
                SELECT *,
                    ROUND(
                        (julianday('now','localtime') -
                         julianday(visit_date || 'T' || entry_time)) * 1440, 1
                    ) AS minutes_so_far
                FROM visitor_sessions
                WHERE visit_date=? AND exit_time IS NULL
                ORDER BY entry_time
            """, (visit_date,))
            return [dict(r) for r in c.fetchall()]

    def get_sessions(self, visit_date: str = None,
                      visitor_id: str = None,
                      from_date: str = None,
                      to_date: str = None,
                      door_id: str = None) -> List[Dict]:
        conds, params = [], []
        if visit_date: conds.append("visit_date=?");   params.append(visit_date)
        if visitor_id: conds.append("visitor_id=?");   params.append(visitor_id)
        if from_date:  conds.append("visit_date>=?");  params.append(from_date)
        if to_date:    conds.append("visit_date<=?");  params.append(to_date)
        if door_id:    conds.append("door_id=?");      params.append(door_id)
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        with self._cur() as c:
            c.execute(
                f"SELECT * FROM visitor_sessions {where} "
                f"ORDER BY visit_date DESC, entry_time DESC",
                params
            )
            return [dict(r) for r in c.fetchall()]

    def get_daily_summary(self, visit_date: str) -> List[Dict]:
        with self._cur() as c:
            c.execute("""
                SELECT
                    visitor_id, name, dept, person_type,
                    COUNT(*)              AS total_visits,
                    SUM(duration_minutes) AS total_minutes,
                    MAX(duration_minutes) AS longest_stay,
                    MIN(entry_time)       AS first_entry,
                    MAX(exit_time)        AS last_exit,
                    SUM(over_15min)       AS over_15_count,
                    MAX(CASE WHEN exit_time IS NULL THEN 1 ELSE 0 END)
                                          AS currently_inside,
                    GROUP_CONCAT(
                        door_id || ':' || entry_time || '→' ||
                        COALESCE(exit_door || ':' || exit_time, '?'),
                        '  |  '
                    )                     AS visit_timeline
                FROM visitor_sessions
                WHERE visit_date=?
                GROUP BY visitor_id
                ORDER BY total_visits DESC
            """, (visit_date,))
            return [dict(r) for r in c.fetchall()]

    def get_door_stats(self, visit_date: str) -> List[Dict]:
        """Traffic per door — useful for understanding HLC usage patterns."""
        with self._cur() as c:
            c.execute("""
                SELECT
                    COALESCE(door_id, 'UNKNOWN')  AS door,
                    COUNT(*)                       AS total_entries,
                    COUNT(DISTINCT visitor_id)     AS unique_visitors,
                    ROUND(AVG(duration_minutes),1) AS avg_duration
                FROM visitor_sessions
                WHERE visit_date=?
                GROUP BY door_id
                ORDER BY total_entries DESC
            """, (visit_date,))
            return [dict(r) for r in c.fetchall()]

    def get_stats(self, visit_date: str) -> Dict:
        with self._cur() as c:
            c.execute("""
                SELECT
                    COUNT(DISTINCT visitor_id)  AS unique_visitors,
                    COUNT(*)                    AS total_visits,
                    SUM(CASE WHEN exit_time IS NULL THEN 1 ELSE 0 END)
                                                AS currently_inside,
                    SUM(over_15min)             AS over_15_count,
                    ROUND(AVG(duration_minutes),1) AS avg_duration_min,
                    ROUND(MAX(duration_minutes),1) AS max_duration_min
                FROM visitor_sessions WHERE visit_date=?
            """, (visit_date,))
            row = dict(c.fetchone() or {})

            # Add door breakdown
            row["door_stats"] = self.get_door_stats(visit_date)
            return row

    def update_purpose(self, session_id: int, purpose: str):
        with self._cur() as c:
            c.execute("UPDATE visitor_sessions SET purpose=? WHERE id=?",
                      (purpose, session_id))

    # ── Alert management ──────────────────────────────────────────────────────

    def sessions_needing_alert(self, threshold_minutes: float = 5.0) -> List[Dict]:
        today = date.today().isoformat()
        with self._cur() as c:
            c.execute("""
                SELECT *,
                    ROUND(
                        (julianday('now','localtime') -
                         julianday(visit_date || 'T' || entry_time)) * 1440, 1
                    ) AS minutes_elapsed
                FROM visitor_sessions
                WHERE visit_date=?
                  AND exit_time IS NULL
                  AND alert_sent=0
                  AND ROUND(
                      (julianday('now','localtime') -
                       julianday(visit_date || 'T' || entry_time)) * 1440, 1
                  ) >= ?
            """, (today, threshold_minutes))
            return [dict(r) for r in c.fetchall()]

    def create_alert(self, session_id: int, visitor_id: str,
                      visitor_name: str, dept: str, visit_date: str,
                      entry_time: str, duration_at_trigger: float) -> int:
        with self._cur() as c:
            c.execute("SELECT id FROM visit_alerts WHERE session_id=?",
                      (session_id,))
            if c.fetchone():
                return -1

            now = datetime.now().strftime("%H:%M:%S")
            c.execute("""
                INSERT INTO visit_alerts
                    (session_id, visitor_id, visitor_name, dept,
                     visit_date, entry_time, triggered_at, duration_at_trigger)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, visitor_id, visitor_name, dept,
                  visit_date, entry_time, now, duration_at_trigger))
            alert_id = c.lastrowid

            c.execute(
                "UPDATE visitor_sessions SET alert_sent=1, alert_id=? WHERE id=?",
                (alert_id, session_id)
            )
            return alert_id

    def log_alert_sent(self, alert_id: int, channel: str,
                        recipient: str, message: str, status: str):
        with self._cur() as c:
            c.execute(
                "INSERT INTO alert_log (alert_id,channel,recipient,message,status) "
                "VALUES (?,?,?,?,?)",
                (alert_id, channel, recipient, message, status)
            )
            col = {"dashboard": "notified_dashboard",
                   "whatsapp":  "notified_whatsapp"}.get(channel.lower())
            if col:
                c.execute(
                    f"UPDATE visit_alerts SET {col}=1 WHERE id=?", (alert_id,)
                )

    def resolve_alert(self, alert_id: int, resolved_by: str, note: str = ""):
        with self._cur() as c:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute("""
                UPDATE visit_alerts
                SET status='RESOLVED', resolved_at=?, resolved_by=?,
                    resolution_note=?
                WHERE id=?
            """, (now, resolved_by, note, alert_id))

    def get_open_alerts(self, visit_date: str = None) -> List[Dict]:
        df = f"AND visit_date='{visit_date}'" if visit_date else ""
        with self._cur() as c:
            c.execute(f"""
                SELECT a.*, s.duration_minutes, s.exit_time as session_exit,
                       s.door_id as entry_door
                FROM visit_alerts a
                LEFT JOIN visitor_sessions s ON a.session_id = s.id
                WHERE a.status='OPEN' {df}
                ORDER BY a.triggered_at DESC
            """)
            return [dict(r) for r in c.fetchall()]

    def get_all_alerts(self, visit_date: str = None,
                        from_date: str = None, to_date: str = None) -> List[Dict]:
        conds, params = [], []
        if visit_date: conds.append("a.visit_date=?");  params.append(visit_date)
        if from_date:  conds.append("a.visit_date>=?"); params.append(from_date)
        if to_date:    conds.append("a.visit_date<=?"); params.append(to_date)
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        with self._cur() as c:
            c.execute(
                f"""SELECT a.*, s.designation, s.person_type AS staff_category
                    FROM visit_alerts a
                    LEFT JOIN visitor_sessions s ON a.session_id = s.id
                    {where}
                    ORDER BY a.triggered_at DESC""",
                params
            )
            return [dict(r) for r in c.fetchall()]
