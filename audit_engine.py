"""
FaceTrack Pro — audit_engine.py
End-of-day audit: re-derives every faculty's attendance from raw
detection timestamps before the record is locked and exported.

Audit statuses:
  CLEAN        stored status matches re-computed — no change
  CORRECTED    auto-fixed unambiguous discrepancy
  FLAGGED      ambiguous or contradictory — needs HR review
  ABSENT_AUTO  no detections at all — auto-inserted as ABSENT
"""

import sqlite3
import os
from datetime import datetime, date
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger

from config import config


@dataclass
class FacultyAuditResult:
    employee_id:      str
    name:             str
    work_date:        str
    first_in:         Optional[str] = None
    last_out:         Optional[str] = None
    first_detection:  Optional[str] = None
    last_detection:   Optional[str] = None
    total_detections: int           = 0
    stored_final:     Optional[str] = None
    audit_morning:    Optional[str] = None
    audit_evening:    Optional[str] = None
    audit_final:      Optional[str] = None
    audit_status:     str           = "UNVERIFIED"
    issues:           List[str]     = field(default_factory=list)
    corrections:      List[str]     = field(default_factory=list)
    flags:            List[str]     = field(default_factory=list)


@dataclass
class DailyAuditReport:
    audit_date:    str
    run_at:        str
    total_faculty: int = 0
    clean:         int = 0
    corrected:     int = 0
    flagged:       int = 0
    absent_auto:   int = 0
    results: List[FacultyAuditResult] = field(default_factory=list)


class AuditEngine:

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.database.db_path
        self._ensure_columns()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_columns(self):
        with self._connect() as conn:
            for col in ("audit_status TEXT DEFAULT 'UNVERIFIED'", "audit_notes TEXT"):
                try:
                    conn.execute(f"ALTER TABLE attendance ADD COLUMN {col}")
                except Exception:
                    pass
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    audit_date    TEXT NOT NULL,
                    run_at        TEXT DEFAULT (datetime('now')),
                    total_faculty INTEGER DEFAULT 0,
                    clean         INTEGER DEFAULT 0,
                    corrected     INTEGER DEFAULT 0,
                    flagged       INTEGER DEFAULT 0,
                    absent_auto   INTEGER DEFAULT 0,
                    triggered_by  TEXT DEFAULT 'system'
                )
            """)

    # ── Main entry point ───────────────────────────────────────

    def run_daily_audit(self, audit_date: str = None,
                        triggered_by: str = "system") -> DailyAuditReport:
        if not audit_date:
            audit_date = date.today().isoformat()

        logger.info(f"Audit starting for {audit_date} (by: {triggered_by})")

        report = DailyAuditReport(
            audit_date=audit_date,
            run_at=datetime.now().isoformat(timespec="seconds"),
        )

        all_faculty       = self._get_all_faculty()
        report.total_faculty = len(all_faculty)

        detections_by_emp = self._get_detections(audit_date)
        stored_by_emp     = self._get_stored(audit_date)

        for fac in all_faculty:
            emp_id = fac["employee_id"]
            result = self._audit_one(
                emp_id        = emp_id,
                name          = fac["name"],
                work_date     = audit_date,
                detections    = detections_by_emp.get(emp_id, []),
                stored_record = stored_by_emp.get(emp_id),
            )
            self._apply_result(result)

            if result.audit_status == "CLEAN":         report.clean       += 1
            elif result.audit_status == "CORRECTED":   report.corrected   += 1
            elif result.audit_status == "FLAGGED":     report.flagged     += 1
            elif result.audit_status == "ABSENT_AUTO": report.absent_auto += 1

            report.results.append(result)

        self._write_log(report, triggered_by)
        logger.success(
            f"Audit done {audit_date}: "
            f"clean={report.clean} corrected={report.corrected} "
            f"flagged={report.flagged} absent_auto={report.absent_auto}"
        )
        return report

    # ── Per-faculty audit ──────────────────────────────────────

    def _audit_one(self, emp_id, name, work_date, detections, stored_record):
        result = FacultyAuditResult(
            employee_id=emp_id, name=name, work_date=work_date
        )
        policy = config.policy

        if stored_record:
            result.stored_final = stored_record.get("final_status")

        # Case 1: no detections at all
        if not detections:
            result.audit_morning = "ABSENT"
            result.audit_evening = "ABSENT"
            result.audit_final   = "ABSENT"
            if stored_record and result.stored_final not in ("ABSENT", None):
                result.corrections.append(
                    f"No detections but stored={result.stored_final} — corrected ABSENT"
                )
                result.audit_status = "CORRECTED"
            elif stored_record:
                result.audit_status = "CLEAN"
            else:
                result.audit_status = "ABSENT_AUTO"
            return result

        # Case 2: has detections
        result.total_detections = len(detections)
        in_times  = sorted([d["detected_at_time"] for d in detections if d["direction"] == "IN"])
        out_times = sorted([d["detected_at_time"] for d in detections if d["direction"] == "OUT"])
        all_times = sorted([d["detected_at_time"] for d in detections])

        result.first_detection = all_times[0]
        result.last_detection  = all_times[-1]
        result.first_in        = in_times[0]  if in_times  else None
        result.last_out        = out_times[-1] if out_times else None

        # Glitch checks
        self._check_glitches(result, detections, all_times)

        # Re-derive morning status
        arrival = result.first_in or result.first_detection
        if arrival < policy.on_time_arrival_cutoff:
            result.audit_morning = "PRESENT"
        elif arrival <= policy.permission_arrival_end:
            result.audit_morning = "PERMISSION"
        elif arrival > policy.afternoon_absent_cutoff:
            result.audit_morning = "ABSENT_MORNING"
        else:
            result.audit_morning = "ABSENT"

        # Re-derive evening status
        departure = result.last_out or result.last_detection
        if departure >= policy.on_time_departure_start:
            result.audit_evening = "PRESENT"
        elif departure >= policy.permission_departure_start:
            result.audit_evening = "PERMISSION"
        elif departure < policy.half_day_cutoff:
            result.audit_evening = "HALF_DAY"
        else:
            result.audit_evening = "ABSENT"

        result.audit_final = self._derive_final(
            result.audit_morning, result.audit_evening, emp_id, work_date[:7]
        )

        # Decide outcome
        if result.flags:
            result.audit_status = "FLAGGED"
        elif not stored_record:
            result.corrections.append(f"Missing record created: {result.audit_final}")
            result.audit_status = "CORRECTED"
        elif result.stored_final == result.audit_final:
            result.audit_status = "CLEAN"
        else:
            result.corrections.append(
                f"{result.stored_final} → {result.audit_final}"
            )
            result.audit_status = "CORRECTED"

        return result

    def _check_glitches(self, result, detections, all_times):
        directions = [d["direction"] for d in detections]

        # All detections same direction — possible camera direction error
        if len(set(directions)) == 1 and len(detections) > 2:
            result.flags.append(
                f"All {len(detections)} detections are '{directions[0]}' — "
                f"possible direction detection error"
            )

        # Gap > 8 hours between consecutive detections
        for i in range(len(all_times) - 1):
            t1  = datetime.strptime(all_times[i],     "%H:%M")
            t2  = datetime.strptime(all_times[i + 1], "%H:%M")
            gap = (t2 - t1).seconds / 3600
            if gap > 8:
                result.flags.append(f"Gap > 8hrs: {all_times[i]} → {all_times[i+1]}")

        # Detection outside reasonable hours
        for t in all_times:
            if t < "06:00" or t > "21:00":
                result.flags.append(f"Detection at unusual time: {t}")

        # Only 1 detection — cannot confirm full session
        if result.total_detections == 1:
            result.flags.append(
                f"Only 1 detection ({all_times[0]}) — cannot confirm full session"
            )

    def _derive_final(self, morning, evening, employee_id, year_month):
        """Mirrors database.py logic — no permission side-effects during audit."""
        policy = config.policy
        if evening == "HALF_DAY":
            return "HALF_DAY"
        if morning == "PRESENT":
            return "FULL_DAY"
        if morning == "PERMISSION":
            count = self._permission_count(employee_id, year_month)
            if count >= policy.max_permissions_per_month:
                return "ABSENT"
            return "FULL_DAY" if evening == "PRESENT" else "PERMISSION"
        if morning == "ABSENT_MORNING":
            return "ABSENT_MORNING"
        return "ABSENT"

    def _permission_count(self, employee_id, year_month):
        with self._connect() as conn:
            row = conn.execute("""
                SELECT permission_count FROM monthly_permissions
                WHERE employee_id=? AND year_month=?
            """, (employee_id, year_month)).fetchone()
            return row["permission_count"] if row else 0

    # ── DB write-back ──────────────────────────────────────────

    def _apply_result(self, result: FacultyAuditResult):
        notes = "; ".join(result.corrections + result.flags + result.issues) or None
        with self._connect() as conn:
            if result.audit_status == "ABSENT_AUTO":
                conn.execute("""
                    INSERT OR IGNORE INTO attendance
                    (employee_id, work_date, morning_status, evening_status,
                     final_status, audit_status, audit_notes)
                    VALUES (?, ?, 'ABSENT', 'ABSENT', 'ABSENT', 'ABSENT_AUTO', ?)
                """, (result.employee_id, result.work_date, notes))
            else:
                conn.execute("""
                    INSERT INTO attendance
                    (employee_id, work_date, morning_status, evening_status,
                     final_status, audit_status, audit_notes, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    ON CONFLICT(employee_id, work_date) DO UPDATE SET
                        morning_status = excluded.morning_status,
                        evening_status = excluded.evening_status,
                        final_status   = excluded.final_status,
                        audit_status   = excluded.audit_status,
                        audit_notes    = excluded.audit_notes,
                        updated_at     = datetime('now')
                """, (
                    result.employee_id, result.work_date,
                    result.audit_morning or "ABSENT",
                    result.audit_evening or "ABSENT",
                    result.audit_final   or "ABSENT",
                    result.audit_status, notes,
                ))

    def _write_log(self, report: DailyAuditReport, triggered_by: str):
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO audit_log
                (audit_date, total_faculty, clean, corrected, flagged, absent_auto, triggered_by)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (report.audit_date, report.total_faculty, report.clean,
                  report.corrected, report.flagged, report.absent_auto, triggered_by))

    # ── Queries ────────────────────────────────────────────────

    def get_audit_summary(self, audit_date: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT * FROM audit_log WHERE audit_date=?
                ORDER BY id DESC LIMIT 1
            """, (audit_date,)).fetchone()
            return dict(row) if row else None

    def get_flagged_records(self, audit_date: str) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT a.*, f.name FROM attendance a
                JOIN faculty f ON a.employee_id = f.employee_id
                WHERE a.work_date=? AND a.audit_status='FLAGGED'
            """, (audit_date,)).fetchall()
            return [dict(r) for r in rows]

    # ── Helpers ────────────────────────────────────────────────

    def _get_all_faculty(self):
        with self._connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM faculty").fetchall()]

    def _get_detections(self, work_date: str) -> Dict[str, List]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM detections WHERE work_date=?", (work_date,)
            ).fetchall()
        out = {}
        for r in rows:
            d = dict(r)
            out.setdefault(d["employee_id"], []).append(d)
        return out

    def _get_stored(self, work_date: str) -> Dict[str, Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM attendance WHERE work_date=?", (work_date,)
            ).fetchall()
        return {dict(r)["employee_id"]: dict(r) for r in rows}


# ── Singleton ─────────────────────────────────────────────────
audit_engine = AuditEngine()
