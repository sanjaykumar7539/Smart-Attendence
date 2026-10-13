"""
FaceTrack Pro — database.py
SQLite attendance database + policy engine + audit-gated CSV export.

Bug fixed:
FIX-DB1: _derive_final() was calling _increment_permission() even on
         the over-quota branch, double-counting usage. Fixed by only
         incrementing when quota is NOT exceeded.
"""

import sqlite3
import os
import csv
import io
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Tuple
from loguru import logger

from config import config


class AttendanceDB:
    """
    SQLite database for FaceTrack Pro.

    Tables:
      faculty              — enrolled staff
      detections           — every raw camera event (permanent log)
      attendance           — one row per (faculty, work_date) — derived status
      monthly_permissions  — running permission quota per faculty per month
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.database.db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_tables()
        logger.info(f"Database ready: {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_tables(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS faculty (
                    employee_id    TEXT PRIMARY KEY,
                    name           TEXT NOT NULL,
                    department     TEXT DEFAULT '',
                    designation    TEXT DEFAULT '',
                    staff_category TEXT DEFAULT 'Teaching',
                    is_hlc         INTEGER DEFAULT 0,
                    face_enrolled  INTEGER DEFAULT 0,
                    faiss_index    INTEGER DEFAULT -1,
                    created_at     TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS detections (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id      TEXT NOT NULL,
                    camera_id        TEXT NOT NULL,
                    direction        TEXT NOT NULL,
                    confidence       REAL,
                    detected_at      TEXT DEFAULT (datetime('now')),
                    work_date        TEXT,
                    detected_at_time TEXT,
                    FOREIGN KEY (employee_id) REFERENCES faculty(employee_id)
                );

                CREATE TABLE IF NOT EXISTS attendance (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id    TEXT NOT NULL,
                    work_date      TEXT NOT NULL,
                    morning_status TEXT DEFAULT 'ABSENT',
                    evening_status TEXT DEFAULT 'ABSENT',
                    final_status   TEXT DEFAULT 'ABSENT',
                    first_seen     TEXT,
                    last_seen      TEXT,
                    notes          TEXT,
                    audit_status   TEXT DEFAULT 'UNVERIFIED',
                    audit_notes    TEXT,
                    updated_at     TEXT DEFAULT (datetime('now')),
                    UNIQUE(employee_id, work_date),
                    FOREIGN KEY (employee_id) REFERENCES faculty(employee_id)
                );

                CREATE TABLE IF NOT EXISTS monthly_permissions (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id      TEXT NOT NULL,
                    year_month       TEXT NOT NULL,
                    permission_count INTEGER DEFAULT 0,
                    UNIQUE(employee_id, year_month),
                    FOREIGN KEY (employee_id) REFERENCES faculty(employee_id)
                );

                CREATE INDEX IF NOT EXISTS idx_det_emp_date
                    ON detections(employee_id, work_date);
                CREATE INDEX IF NOT EXISTS idx_att_emp_date
                    ON attendance(employee_id, work_date);
            """)
            # Migration: add columns if this faculty table pre-dates them
            existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(faculty)")}
            if "staff_category" not in existing_cols:
                conn.execute("ALTER TABLE faculty ADD COLUMN staff_category TEXT DEFAULT 'Teaching'")
                logger.info("Migrated faculty table: added staff_category column")
            if "is_hlc" not in existing_cols:
                conn.execute("ALTER TABLE faculty ADD COLUMN is_hlc INTEGER DEFAULT 0")
                logger.info("Migrated faculty table: added is_hlc column")

    # ── Faculty ────────────────────────────────────────────────

    def add_faculty(self, employee_id: str, name: str,
                    department: str = "", designation: str = "",
                    staff_category: str = "Teaching", is_hlc: bool = False):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO faculty
                (employee_id, name, department, designation, staff_category, is_hlc)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(employee_id) DO UPDATE SET
                    name=excluded.name, department=excluded.department,
                    designation=excluded.designation,
                    staff_category=excluded.staff_category, is_hlc=excluded.is_hlc
            """, (employee_id, name, department, designation, staff_category, int(is_hlc)))

    def set_embedding_index(self, employee_id: str, faiss_idx: int):
        with self._connect() as conn:
            conn.execute("""
                UPDATE faculty SET face_enrolled=1, faiss_index=?
                WHERE employee_id=?
            """, (faiss_idx, employee_id))

    def get_faculty(self, employee_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM faculty WHERE employee_id=?", (employee_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_all_faculty(self) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM faculty ORDER BY name"
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Detections → Attendance ────────────────────────────────

    def log_detection(self, employee_id: str, camera_id: str,
                      direction: str, confidence: float):
        """Log a raw detection event and update today's attendance record."""
        now       = datetime.now()
        work_date = now.strftime("%Y-%m-%d")
        det_time  = now.strftime("%H:%M")

        with self._connect() as conn:
            conn.execute("""
                INSERT INTO detections
                (employee_id, camera_id, direction, confidence, work_date, detected_at_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (employee_id, camera_id, direction, confidence, work_date, det_time))

        self._update_attendance(employee_id, work_date, det_time)

    def log_presence(self, employee_id: str, camera_id: str, confidence: float):
        """Simple classroom presence; no arrival/departure policy or permission quota."""
        now = datetime.now()
        work_date, det_time = now.strftime("%Y-%m-%d"), now.strftime("%H:%M")
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO detections
                (employee_id, camera_id, direction, confidence, work_date, detected_at_time)
                VALUES (?, ?, 'IN', ?, ?, ?)
            """, (employee_id, camera_id, confidence, work_date, det_time))
            conn.execute("""
                INSERT INTO attendance
                (employee_id, work_date, morning_status, evening_status, final_status,
                 first_seen, last_seen, notes)
                VALUES (?, ?, 'PRESENT', 'PRESENT', 'PRESENT', ?, ?, 'Classroom presence mode')
                ON CONFLICT(employee_id, work_date) DO UPDATE SET
                    morning_status='PRESENT', evening_status='PRESENT', final_status='PRESENT',
                    last_seen=excluded.last_seen, notes=excluded.notes,
                    updated_at=datetime('now')
            """, (employee_id, work_date, det_time, det_time))

    def _update_attendance(self, employee_id: str, work_date: str, det_time: str):
        """Re-derive attendance status from all detections so far today."""
        policy = config.policy

        with self._connect() as conn:
            rec = conn.execute("""
                SELECT first_seen, last_seen FROM attendance
                WHERE employee_id=? AND work_date=?
            """, (employee_id, work_date)).fetchone()

            first_seen = rec["first_seen"] if rec and rec["first_seen"] else det_time
            last_seen  = max(rec["last_seen"] if rec and rec["last_seen"] else det_time, det_time)

        # Morning status
        if first_seen <= policy.on_time_arrival_cutoff:
            morning = "PRESENT"
        elif first_seen <= policy.permission_arrival_end:
            morning = "PERMISSION"
        elif first_seen > policy.afternoon_absent_cutoff:
            morning = "ABSENT_MORNING"
        else:
            morning = "ABSENT"

        # Evening status
        if last_seen >= policy.on_time_departure_start:
            evening = "PRESENT"
        elif last_seen >= policy.permission_departure_start:
            evening = "PERMISSION"
        elif last_seen < policy.half_day_cutoff:
            evening = "HALF_DAY"
        else:
            evening = "ABSENT"

        final = self._derive_final(morning, evening, employee_id, work_date[:7])

        with self._connect() as conn:
            conn.execute("""
                INSERT INTO attendance
                (employee_id, work_date, morning_status, evening_status,
                 final_status, first_seen, last_seen, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(employee_id, work_date) DO UPDATE SET
                    morning_status = excluded.morning_status,
                    evening_status = excluded.evening_status,
                    final_status   = excluded.final_status,
                    first_seen     = excluded.first_seen,
                    last_seen      = excluded.last_seen,
                    updated_at     = datetime('now')
            """, (employee_id, work_date, morning, evening, final, first_seen, last_seen))

    def _try_consume_permission(self, employee_id: str, year_month: str) -> bool:
        """
        Attempt to use one permission credit for this month.
        Returns True and increments if quota available, False if exhausted
        (does NOT increment on the exhausted path — FIX-DB1).
        """
        policy = config.policy
        if self._permission_count(employee_id, year_month) >= policy.max_permissions_per_month:
            return False
        self._increment_permission(employee_id, year_month)
        return True

    def _derive_final(self, morning: str, evening: str,
                      employee_id: str, year_month: str) -> str:
        """
        EGS Pillay attendance policy decision matrix.

        Sessions: forenoon 09:10-12:35, afternoon 13:20-16:10.
        Permission: 45min grace on the morning arrival OR the evening
        departure — max 2/month, one shared quota regardless of which side
        used it. Using the grace window on either side always costs exactly
        one credit for that day, even if both sides used it.
        Half-day: attended one full session (forenoon OR afternoon) but not
        the other — works in both directions.
        """
        # Missed forenoon entirely
        if morning == "ABSENT_MORNING":
            if evening in ("PRESENT", "PERMISSION"):
                # Afternoon-only half day
                if evening == "PERMISSION" and not self._try_consume_permission(employee_id, year_month):
                    return "ABSENT"
                return "HALF_DAY"
            return "ABSENT"   # missed forenoon AND didn't stay for afternoon either

        # Attended forenoon (on-time or via permission) but left before the
        # afternoon session started — forenoon-only half day
        if evening == "HALF_DAY":
            if morning == "PERMISSION" and not self._try_consume_permission(employee_id, year_month):
                return "ABSENT"
            return "HALF_DAY"

        # On-time arrival
        if morning == "PRESENT":
            if evening == "PERMISSION":
                if not self._try_consume_permission(employee_id, year_month):
                    return "ABSENT"
                return "FULL_DAY"
            return "FULL_DAY"   # evening PRESENT or ABSENT

        # Late arrival via permission grace
        if morning == "PERMISSION":
            if evening == "PRESENT":
                if not self._try_consume_permission(employee_id, year_month):
                    return "ABSENT"
                return "FULL_DAY"
            # evening PERMISSION or ABSENT — grace used on arrival, and
            # departure wasn't a clean on-time PRESENT either. Distinct
            # PERMISSION-day status (not a full day, but not absent).
            if not self._try_consume_permission(employee_id, year_month):
                return "ABSENT"
            return "PERMISSION"

        return "ABSENT"

    def _permission_count(self, employee_id: str, year_month: str) -> int:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT permission_count FROM monthly_permissions
                WHERE employee_id=? AND year_month=?
            """, (employee_id, year_month)).fetchone()
            return row["permission_count"] if row else 0

    def _increment_permission(self, employee_id: str, year_month: str):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO monthly_permissions (employee_id, year_month, permission_count)
                VALUES (?, ?, 1)
                ON CONFLICT(employee_id, year_month) DO UPDATE SET
                    permission_count = permission_count + 1
            """, (employee_id, year_month))

    # ── Queries ────────────────────────────────────────────────

    def get_today_attendance(self) -> List[Dict]:
        today = date.today().isoformat()
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT a.*, f.name, f.department, f.designation
                FROM attendance a
                JOIN faculty f ON a.employee_id = f.employee_id
                WHERE a.work_date = ?
                ORDER BY a.first_seen ASC NULLS LAST
            """, (today,)).fetchall()
        return [dict(r) for r in rows]

    def reset_presence(self, scope: str) -> Dict:
        """Back up the student DB before clearing a date-scoped attendance history."""
        if scope not in ('today', 'previous'):
            raise ValueError('Unsupported reset scope')
        from pathlib import Path
        from contextlib import closing
        today = date.today().isoformat()
        operator = '=' if scope == 'today' else '<'
        backup_dir = Path(self.db_path).parent / 'backups'
        backup_dir.mkdir(exist_ok=True)
        backup_path = backup_dir / f'attendance_{datetime.now():%Y%m%d_%H%M%S_%f}.db'
        with closing(self._connect()) as conn:
            with closing(sqlite3.connect(str(backup_path))) as backup:
                conn.backup(backup)
            with conn:
                detections = conn.execute(
                    f'DELETE FROM detections WHERE work_date {operator} ?', (today,)).rowcount
                attendance = conn.execute(
                    f'DELETE FROM attendance WHERE work_date {operator} ?', (today,)).rowcount
        return {'scope': scope, 'attendance_deleted': attendance,
                'detections_deleted': detections, 'backup_saved': True}

    def get_today_stats(self) -> Dict:
        today     = date.today().isoformat()
        all_fac   = self.get_all_faculty()
        today_att = self.get_today_attendance()

        status_map = {r["employee_id"]: r["final_status"] for r in today_att}
        stats = {"total": len(all_fac), "present": 0, "permission": 0,
                 "half_day": 0, "absent": 0, "date": today}

        for f in all_fac:
            st = status_map.get(f["employee_id"], "ABSENT")
            if   st in ("PRESENT", "FULL_DAY"): stats["present"]    += 1
            elif st == "PERMISSION":            stats["permission"] += 1
            elif st == "HALF_DAY":              stats["half_day"]   += 1
            else:                               stats["absent"]     += 1

        return stats

    # ── Export ─────────────────────────────────────────────────

    def export_csv(self, from_date: str, to_date: str,
                   employee_id: str = None,
                   force: bool = False) -> Tuple[Optional[str], List[str]]:
        """
        Export attendance CSV.
        Returns (csv_string, warnings).
        Blocks if audit not run unless force=True.
        """
        warnings = []

        with self._connect() as conn:
            unaudited = conn.execute("""
                SELECT COUNT(*) as cnt FROM attendance
                WHERE work_date BETWEEN ? AND ?
                  AND audit_status = 'UNVERIFIED'
            """, (from_date, to_date)).fetchone()["cnt"]

        if unaudited > 0 and not force:
            warnings.append(
                f"{unaudited} records unaudited for {from_date}–{to_date}. "
                f"Run end-of-day audit first, or use force=True to bypass."
            )
            return None, warnings

        query  = """
            SELECT a.work_date, a.employee_id, f.name, f.department, f.designation,
                   a.first_seen, a.last_seen, a.morning_status,
                   a.evening_status, a.final_status, a.audit_status
            FROM attendance a
            JOIN faculty f ON a.employee_id = f.employee_id
            WHERE a.work_date BETWEEN ? AND ?
        """
        params = [from_date, to_date]
        if employee_id:
            query  += " AND a.employee_id = ?"
            params.append(employee_id)
        query += " ORDER BY a.work_date, f.name"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([f"# {config.college_name} — ATTENDANCE REPORT"])
        writer.writerow([f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
        writer.writerow([f"# Period: {from_date} to {to_date}"])
        writer.writerow([])
        writer.writerow([
            "Date", "Employee ID", "Name", "Department", "Designation",
            "Arrival", "Departure", "Morning", "Evening", "Final Status", "Audit",
        ])
        for r in rows:
            writer.writerow([
                r["work_date"], r["employee_id"], r["name"],
                r["department"], r["designation"],
                r["first_seen"] or "—", r["last_seen"] or "—",
                r["morning_status"], r["evening_status"],
                r["final_status"], r["audit_status"],
            ])

        return output.getvalue(), warnings

    def export_detections_csv(self, from_date: str, to_date: str) -> str:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT d.work_date, d.detected_at_time, d.employee_id,
                       f.name, d.direction, d.confidence, d.camera_id
                FROM detections d
                JOIN faculty f ON d.employee_id = f.employee_id
                WHERE d.work_date BETWEEN ? AND ?
                ORDER BY d.work_date, d.detected_at
            """, (from_date, to_date)).fetchall()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Date", "Time", "Employee ID", "Name",
                         "Direction", "Confidence", "Camera"])
        for r in rows:
            writer.writerow([
                r["work_date"], r["detected_at_time"], r["employee_id"],
                r["name"], r["direction"], f"{r['confidence']:.2f}", r["camera_id"],
            ])
        return output.getvalue()

    def export_monthly_csv(self, year_month: str) -> str:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT f.employee_id, f.name, f.department,
                    SUM(CASE WHEN a.final_status IN ('PRESENT','FULL_DAY') THEN 1 ELSE 0 END) AS present_days,
                    SUM(CASE WHEN a.final_status = 'PERMISSION'             THEN 1 ELSE 0 END) AS permission_days,
                    SUM(CASE WHEN a.final_status = 'HALF_DAY'               THEN 1 ELSE 0 END) AS half_days,
                    SUM(CASE WHEN a.final_status IN ('ABSENT','ABSENT_MORNING') THEN 1 ELSE 0 END) AS absent_days,
                    COUNT(a.id) AS total_days
                FROM faculty f
                LEFT JOIN attendance a
                    ON f.employee_id = a.employee_id
                    AND substr(a.work_date,1,7) = ?
                GROUP BY f.employee_id
                ORDER BY f.name
            """, (year_month,)).fetchall()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([f"# Monthly Report — {year_month} | {config.college_name}"])
        writer.writerow([])
        writer.writerow(["Employee ID", "Name", "Department",
                         "Present", "Permission", "Half Day", "Absent", "Working Days"])
        for r in rows:
            writer.writerow([
                r["employee_id"], r["name"], r["department"],
                r["present_days"], r["permission_days"],
                r["half_days"], r["absent_days"], r["total_days"],
            ])
        return output.getvalue()


# ── Singleton ─────────────────────────────────────────────────
db = AttendanceDB()
