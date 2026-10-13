"""
FaceTrack Pro — hr_database.py
Leave management, On-Duty applications, manual attendance overrides.
"""

import sqlite3
import os
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict
from loguru import logger

from config import config


LEAVE_LABELS = {
    "CL":   "Casual Leave",
    "ML":   "Medical / Sick Leave",
    "EL":   "Earned Leave",
    "MAT":  "Maternity Leave",
    "PAT":  "Paternity Leave",
    "COMP": "Compensatory Off",
    "OD":   "On-Duty",
    "LOP":  "Loss of Pay",
    "SL":   "Special Leave",
}

LEAVE_ENTITLEMENTS = {
    "CL": 12, "ML": 12, "EL": 30,
    "COMP": 0, "OD": 0, "LOP": 0,
}


class HRDatabase:

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.database.db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_tables(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS leave_applications (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id    TEXT NOT NULL,
                    leave_type     TEXT NOT NULL,
                    from_date      TEXT NOT NULL,
                    to_date        TEXT NOT NULL,
                    days           REAL NOT NULL,
                    reason         TEXT,
                    document_ref   TEXT,
                    status         TEXT DEFAULT 'PENDING',
                    applied_by     TEXT,
                    approved_by    TEXT,
                    applied_at     TEXT DEFAULT (datetime('now')),
                    approved_at    TEXT,
                    rejection_note TEXT
                );

                CREATE TABLE IF NOT EXISTS od_applications (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id     TEXT NOT NULL,
                    od_date         TEXT NOT NULL,
                    half_day        INTEGER DEFAULT 0,
                    session         TEXT DEFAULT 'FULL',
                    purpose         TEXT NOT NULL,
                    destination     TEXT,
                    expected_return TEXT,
                    status          TEXT DEFAULT 'PENDING',
                    approved_by     TEXT,
                    applied_at      TEXT DEFAULT (datetime('now')),
                    approved_at     TEXT,
                    remarks         TEXT
                );

                CREATE TABLE IF NOT EXISTS attendance_overrides (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id     TEXT NOT NULL,
                    work_date       TEXT NOT NULL,
                    original_status TEXT,
                    override_status TEXT NOT NULL,
                    override_reason TEXT NOT NULL,
                    override_type   TEXT DEFAULT 'MANUAL',
                    reference_id    INTEGER,
                    overridden_by   TEXT NOT NULL,
                    overridden_at   TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS leave_balances (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id  TEXT NOT NULL,
                    leave_type   TEXT NOT NULL,
                    year         INTEGER NOT NULL,
                    entitled     REAL DEFAULT 0,
                    used         REAL DEFAULT 0,
                    pending      REAL DEFAULT 0,
                    carried_fwd  REAL DEFAULT 0,
                    UNIQUE(employee_id, leave_type, year)
                );

                CREATE INDEX IF NOT EXISTS idx_leave_emp ON leave_applications(employee_id, from_date);
                CREATE INDEX IF NOT EXISTS idx_od_emp    ON od_applications(employee_id, od_date);
                CREATE INDEX IF NOT EXISTS idx_override  ON attendance_overrides(employee_id, work_date);
            """)

    # ── Leave ──────────────────────────────────────────────────

    def apply_leave(self, employee_id: str, leave_type: str,
                    from_date: str, to_date: str, days: float,
                    reason: str = "", document_ref: str = "") -> Dict:
        bal = self.get_leave_balance(employee_id, leave_type)
        ent = LEAVE_ENTITLEMENTS.get(leave_type, 0)
        if ent > 0 and bal.get("balance", ent) < days:
            return {
                "success": False,
                "error": (
                    f"Insufficient {LEAVE_LABELS.get(leave_type, leave_type)} balance. "
                    f"Available: {bal.get('balance', 0):.1f} days, Requested: {days}"
                ),
            }

        with self._connect() as conn:
            cur = conn.execute("""
                INSERT INTO leave_applications
                (employee_id, leave_type, from_date, to_date, days, reason,
                 document_ref, applied_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (employee_id, leave_type, from_date, to_date, days,
                  reason, document_ref, employee_id))
            app_id = cur.lastrowid
        self._adjust_pending(employee_id, leave_type, days, add=True)
        return {"success": True, "application_id": app_id}

    def approve_leave(self, application_id: int, approved_by: str,
                      remarks: str = "") -> Dict:
        with self._connect() as conn:
            app = conn.execute(
                "SELECT * FROM leave_applications WHERE id=?", (application_id,)
            ).fetchone()
            if not app:
                return {"success": False, "error": "Application not found"}
            if app["status"] != "PENDING":
                return {"success": False, "error": f"Already {app['status']}"}
            app = dict(app)
            conn.execute("""
                UPDATE leave_applications
                SET status='APPROVED', approved_by=?,
                    approved_at=datetime('now'), rejection_note=?
                WHERE id=?
            """, (approved_by, remarks, application_id))

        self._apply_leave_to_attendance(
            app["employee_id"], app["leave_type"],
            app["from_date"], app["to_date"], application_id, approved_by,
        )
        self._adjust_pending(app["employee_id"], app["leave_type"], app["days"], add=False)
        self._deduct_balance(app["employee_id"], app["leave_type"], app["days"])
        return {"success": True}

    def reject_leave(self, application_id: int, rejected_by: str,
                     reason: str) -> Dict:
        with self._connect() as conn:
            app = conn.execute(
                "SELECT * FROM leave_applications WHERE id=?", (application_id,)
            ).fetchone()
            if not app:
                return {"success": False, "error": "Application not found"}
            conn.execute("""
                UPDATE leave_applications
                SET status='REJECTED', approved_by=?,
                    approved_at=datetime('now'), rejection_note=?
                WHERE id=?
            """, (rejected_by, reason, application_id))
        self._adjust_pending(app["employee_id"], app["leave_type"], app["days"], add=False)
        return {"success": True}

    def _apply_leave_to_attendance(self, employee_id, leave_type, from_date,
                                    to_date, app_id, approved_by):
        start  = datetime.strptime(from_date, "%Y-%m-%d").date()
        end    = datetime.strptime(to_date,   "%Y-%m-%d").date()
        status = f"LEAVE_{leave_type}"

        current = start
        while current <= end:
            date_str = current.isoformat()
            with self._connect() as conn:
                existing = conn.execute(
                    "SELECT final_status FROM attendance WHERE employee_id=? AND work_date=?",
                    (employee_id, date_str)
                ).fetchone()
                original = existing["final_status"] if existing else "ABSENT"
                conn.execute("""
                    INSERT INTO attendance
                    (employee_id, work_date, final_status, morning_status,
                     evening_status, notes, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                    ON CONFLICT(employee_id, work_date) DO UPDATE SET
                        final_status   = excluded.final_status,
                        morning_status = excluded.morning_status,
                        evening_status = excluded.evening_status,
                        notes          = excluded.notes,
                        updated_at     = datetime('now')
                """, (employee_id, date_str, status, status, status,
                      f"Leave approval #{app_id}"))
                conn.execute("""
                    INSERT INTO attendance_overrides
                    (employee_id, work_date, original_status, override_status,
                     override_reason, override_type, reference_id, overridden_by)
                    VALUES (?, ?, ?, ?, ?, 'LEAVE', ?, ?)
                """, (employee_id, date_str, original, status,
                      LEAVE_LABELS.get(leave_type, leave_type), app_id, approved_by))
            current += timedelta(days=1)

    # ── On-Duty ────────────────────────────────────────────────

    def apply_od(self, employee_id: str, od_date: str, purpose: str,
                 destination: str = "", session: str = "FULL",
                 expected_return: str = "") -> Dict:
        with self._connect() as conn:
            cur = conn.execute("""
                INSERT INTO od_applications
                (employee_id, od_date, half_day, session, purpose, destination, expected_return)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (employee_id, od_date,
                  1 if session in ("AM", "PM") else 0,
                  session, purpose, destination, expected_return))
        return {"success": True, "application_id": cur.lastrowid}

    def approve_od(self, application_id: int, approved_by: str,
                   remarks: str = "") -> Dict:
        with self._connect() as conn:
            app = conn.execute(
                "SELECT * FROM od_applications WHERE id=?", (application_id,)
            ).fetchone()
            if not app:
                return {"success": False, "error": "OD application not found"}
            app = dict(app)
            conn.execute("""
                UPDATE od_applications
                SET status='APPROVED', approved_by=?,
                    approved_at=datetime('now'), remarks=?
                WHERE id=?
            """, (approved_by, remarks, application_id))

        od_map    = {"FULL": "OD_FULL", "AM": "OD_AM", "PM": "OD_PM"}
        od_status = od_map.get(app["session"], "OD_FULL")

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT final_status FROM attendance WHERE employee_id=? AND work_date=?",
                (app["employee_id"], app["od_date"])
            ).fetchone()
            original = existing["final_status"] if existing else "ABSENT"
            conn.execute("""
                INSERT INTO attendance
                (employee_id, work_date, final_status, morning_status,
                 evening_status, notes, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(employee_id, work_date) DO UPDATE SET
                    final_status = excluded.final_status,
                    notes        = excluded.notes,
                    updated_at   = datetime('now')
            """, (app["employee_id"], app["od_date"], od_status, od_status, od_status,
                  f"OD: {app['purpose']}"))
            conn.execute("""
                INSERT INTO attendance_overrides
                (employee_id, work_date, original_status, override_status,
                 override_reason, override_type, reference_id, overridden_by)
                VALUES (?, ?, ?, ?, ?, 'OD', ?, ?)
            """, (app["employee_id"], app["od_date"], original, od_status,
                  app["purpose"], application_id, approved_by))
        return {"success": True}

    # ── Manual override ────────────────────────────────────────

    def manual_override(self, employee_id: str, work_date: str,
                        new_status: str, reason: str, overridden_by: str,
                        first_seen: str = None, last_seen: str = None) -> Dict:
        if not reason or len(reason.strip()) < 5:
            return {"success": False, "error": "Reason is required (minimum 5 characters)"}

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM attendance WHERE employee_id=? AND work_date=?",
                (employee_id, work_date)
            ).fetchone()
            original = existing["final_status"] if existing else "NO_RECORD"

            conn.execute("""
                INSERT INTO attendance
                (employee_id, work_date, final_status, morning_status,
                 evening_status, first_seen, last_seen, notes, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(employee_id, work_date) DO UPDATE SET
                    final_status   = excluded.final_status,
                    morning_status = excluded.morning_status,
                    evening_status = excluded.evening_status,
                    first_seen     = COALESCE(excluded.first_seen, first_seen),
                    last_seen      = COALESCE(excluded.last_seen, last_seen),
                    notes          = excluded.notes,
                    updated_at     = datetime('now')
            """, (employee_id, work_date, new_status, new_status, new_status,
                  first_seen, last_seen, f"HR override: {reason}"))

            conn.execute("""
                INSERT INTO attendance_overrides
                (employee_id, work_date, original_status, override_status,
                 override_reason, overridden_by)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (employee_id, work_date, original, new_status, reason, overridden_by))

        return {"success": True, "changed_from": original, "changed_to": new_status}

    # ── Queries ────────────────────────────────────────────────

    def get_pending_applications(self) -> Dict:
        with self._connect() as conn:
            leaves = conn.execute(
                "SELECT * FROM leave_applications WHERE status='PENDING' ORDER BY applied_at"
            ).fetchall()
            ods = conn.execute(
                "SELECT * FROM od_applications WHERE status='PENDING' ORDER BY applied_at"
            ).fetchall()
        return {"leave": [dict(r) for r in leaves], "od": [dict(r) for r in ods]}

    def get_override_history(self, employee_id: str = None,
                              work_date: str = None) -> List[Dict]:
        q = "SELECT * FROM attendance_overrides WHERE 1=1"
        p = []
        if employee_id:
            q += " AND employee_id=?"; p.append(employee_id)
        if work_date:
            q += " AND work_date=?";   p.append(work_date)
        q += " ORDER BY overridden_at DESC"
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(q, p).fetchall()]

    def get_leave_balance(self, employee_id: str, leave_type: str = None,
                           year: int = None) -> Dict:
        yr = year or date.today().year
        with self._connect() as conn:
            if leave_type:
                row = conn.execute("""
                    SELECT * FROM leave_balances
                    WHERE employee_id=? AND leave_type=? AND year=?
                """, (employee_id, leave_type, yr)).fetchone()
                if row:
                    r = dict(row)
                    r["balance"] = r["entitled"] + r["carried_fwd"] - r["used"]
                    return r
                ent = LEAVE_ENTITLEMENTS.get(leave_type, 0)
                return {"entitled": float(ent), "used": 0.0, "pending": 0.0,
                        "carried_fwd": 0.0, "balance": float(ent)}
            else:
                rows = conn.execute("""
                    SELECT * FROM leave_balances WHERE employee_id=? AND year=?
                """, (employee_id, yr)).fetchall()
                result = {}
                for r in rows:
                    d = dict(r)
                    d["balance"] = d["entitled"] + d["carried_fwd"] - d["used"]
                    result[d["leave_type"]] = d
                return result

    def get_all_leave_balances(self, year: int = None) -> List[Dict]:
        yr = year or date.today().year
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT f.employee_id, f.name, f.department, lb.*
                FROM faculty f
                LEFT JOIN leave_balances lb
                    ON f.employee_id = lb.employee_id AND lb.year = ?
                ORDER BY f.name
            """, (yr,)).fetchall()
        return [dict(r) for r in rows]

    def initialize_leave_balances(self, employee_id: str, year: int = None):
        yr = year or date.today().year
        with self._connect() as conn:
            for lt, ent in LEAVE_ENTITLEMENTS.items():
                if ent > 0:
                    conn.execute("""
                        INSERT OR IGNORE INTO leave_balances
                        (employee_id, leave_type, year, entitled)
                        VALUES (?, ?, ?, ?)
                    """, (employee_id, lt, yr, ent))

    def get_faculty_leave_history(self, employee_id: str, year: int = None) -> Dict:
        yr = str(year or date.today().year)
        with self._connect() as conn:
            leaves = conn.execute("""
                SELECT * FROM leave_applications
                WHERE employee_id=? AND substr(from_date,1,4)=?
                ORDER BY from_date DESC
            """, (employee_id, yr)).fetchall()
            ods = conn.execute("""
                SELECT * FROM od_applications
                WHERE employee_id=? AND substr(od_date,1,4)=?
                ORDER BY od_date DESC
            """, (employee_id, yr)).fetchall()
            overrides = conn.execute("""
                SELECT * FROM attendance_overrides
                WHERE employee_id=? AND substr(work_date,1,4)=?
                ORDER BY overridden_at DESC
            """, (employee_id, yr)).fetchall()
        return {
            "leaves":    [dict(r) for r in leaves],
            "od":        [dict(r) for r in ods],
            "overrides": [dict(r) for r in overrides],
        }

    # ── Internal helpers ───────────────────────────────────────

    def _adjust_pending(self, employee_id, leave_type, days, add=True):
        yr    = date.today().year
        delta = days if add else -days
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO leave_balances (employee_id, leave_type, year, pending)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(employee_id, leave_type, year) DO UPDATE SET
                    pending = MAX(0, pending + ?)
            """, (employee_id, leave_type, yr, max(0.0, delta), delta))

    def _deduct_balance(self, employee_id, leave_type, days):
        yr = date.today().year
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO leave_balances (employee_id, leave_type, year, used)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(employee_id, leave_type, year) DO UPDATE SET
                    used = used + ?
            """, (employee_id, leave_type, yr, days, days))


# ── Singleton ─────────────────────────────────────────────────
hr_db = HRDatabase()
