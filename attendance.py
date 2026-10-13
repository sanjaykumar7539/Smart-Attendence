"""
attendance.py
──────────────
Main faculty attendance processor.
Handles ALL detections from the primary staff-room camera (CAM-01).

Responsibilities:
  - Policy engine    : on-time / permission / half-day / absent
  - Morning status   : first detection of the day
  - Evening status   : last detection before cut-off
  - Final status     : resolved from morning + evening
  - VoteBuffer       : multi-frame confirmation before writing DB
  - process_detection(): single entry point called by main.py

Does NOT handle:
  - HLC cameras      → hlc_processor_v4.py
  - Any future rooms → their own processor modules

Policy (EGS Pillay):
  Arrival  ≤ 09:05            → PRESENT
  Arrival  09:05 – 09:55      → PERMISSION  (max 2/month)
  Arrival  > 09:55            → ABSENT_MORNING
  Departure < 12:45           → HALF_DAY
  Departure ≥ 16:10           → PRESENT (full day confirmed)
"""

import threading
import logging
from datetime import datetime, date
from collections import deque
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class VoteBuffer:
    """
    Accumulates N recognition votes for a track_id before accepting identity.
    Prevents single-frame false positives from triggering attendance.

    confirm_n:      votes needed to confirm (config.model.vote_confirm_n)
    window_size:    rolling window to count votes within (config.model.vote_window_size)
    """

    def __init__(self, confirm_n: int = 2, window_size: int = 10):
        self.confirm_n   = confirm_n
        self.window_size = window_size
        # track_id → deque of employee_id strings
        self._votes: Dict[int, deque] = {}

    def vote(self, track_id: int, employee_id: str) -> Optional[str]:
        """
        Add a vote for employee_id on this track_id.
        Returns confirmed employee_id if threshold reached, else None.
        """
        if track_id not in self._votes:
            self._votes[track_id] = deque(maxlen=self.window_size)

        self._votes[track_id].append(employee_id)
        votes = list(self._votes[track_id])

        # Count votes for each candidate in this window
        counts: Dict[str, int] = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1

        best_id    = max(counts, key=counts.get)
        best_count = counts[best_id]

        if best_count >= self.confirm_n:
            return best_id
        return None

    def clear(self, track_id: int):
        self._votes.pop(track_id, None)

    def clear_all(self):
        self._votes.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Main attendance processor
# ─────────────────────────────────────────────────────────────────────────────

class AttendanceProcessor:
    """
    Main faculty attendance processor.
    One instance shared across all threads.
    Thread-safe via internal lock.

    Policy engine (on-time/permission/half-day/absent, final status,
    permission quota) lives in ONE place: database.py's log_detection() /
    _update_attendance() / _derive_final(). This class does NOT reimplement
    that logic — it only handles multi-frame vote confirmation and name
    caching, then delegates the actual attendance write to db.log_detection(),
    which already re-derives first_seen/last_seen/morning/evening/final
    status from all of today's detections in one call.

    Usage in main.py:
        attendance = AttendanceProcessor(db, config)
        ...
        result = attendance.process_detection(detection_dict)
    """

    def __init__(self, db, config):
        self.db     = db
        self.config = config
        self.votes  = VoteBuffer(
            confirm_n   = config.model.vote_confirm_n,
            window_size = config.model.vote_window_size,
        )
        self._lock = threading.Lock()

        # Cache: employee_id → name (avoid repeated DB lookups)
        self._name_cache: Dict[str, str] = {}

        logger.info("AttendanceProcessor initialised (main faculty camera)")

    # ── Public entry point ────────────────────────────────────────────────────

    def process_detection(self, detection: Dict[str, Any]) -> Optional[Dict]:
        """
        Called by main.py for every confirmed face crossing.

        detection dict:
            employee_id    str
            name           str
            direction      str   "IN" | "OUT"
            confidence     float  0.0 – 1.0
            track_id       int
            time           str   "HH:MM:SS"
            camera_id      str

        Returns dict with:
            type           "attendance"
            employee_id    str
            name           str
            direction      str
            time           str
            confidence     str  e.g. "94%"
            status         str  final attendance status (from database.py)
            action         str  "created" | "updated"
        Or None if detection should be ignored.
        """
        employee_id = detection.get("employee_id")
        if not employee_id or employee_id == "Unknown":
            return None

        name       = detection.get("name") or self._resolve_name(employee_id)
        direction  = detection.get("direction", "IN")
        confidence = float(detection.get("confidence", 0))
        time_str   = detection.get("time", datetime.now().strftime("%H:%M:%S"))
        camera_id  = detection.get("camera_id", "CAM-01")
        today      = date.today().isoformat()

        with self._lock:
            return self._handle(
                employee_id, name, direction,
                confidence, time_str, camera_id, today
            )

    # ── Internal logic ────────────────────────────────────────────────────────

    def _handle(self, employee_id: str, name: str, direction: str,
                 confidence: float, time_str: str, camera_id: str,
                 today: str) -> Dict:
        """
        Single write path for both IN and OUT — database.py's log_detection()
        logs the raw event AND re-derives the full day's status (morning /
        evening / final / permission quota) from all of today's detections,
        so there's no separate entry/exit branching needed here.
        """
        existing = self._get_today_record(employee_id, today)
        action   = "updated" if existing else "created"

        if self.config.tracker.attendance_mode == "presence":
            self.db.log_presence(employee_id, camera_id, confidence)
        else:
            self.db.log_detection(employee_id, camera_id, direction, confidence)

        updated = self._get_today_record(employee_id, today)
        status  = updated.get("final_status", "PRESENT") if updated else "PRESENT"

        logger.info(
            f"{direction} | {name} ({employee_id}) | {time_str} | "
            f"status={status} | conf={confidence:.0%}"
        )
        return {
            "type":        "attendance",
            "employee_id": employee_id,
            "name":        name,
            "direction":   direction,
            "time":        time_str,
            "confidence":  f"{confidence:.0%}",
            "status":      status,
            "action":      action,
        }

    # ── DB helper wrappers ────────────────────────────────────────────────────

    def _get_today_record(self, employee_id: str,
                           today: str) -> Optional[Dict]:
        """Get today's attendance row for this employee."""
        try:
            rows = self.db.get_today_attendance()
            for r in rows:
                if r["employee_id"] == employee_id:
                    return r
        except Exception as e:
            logger.warning(f"_get_today_record error: {e}")
        return None

    def _resolve_name(self, employee_id: str) -> str:
        if employee_id in self._name_cache:
            return self._name_cache[employee_id]
        try:
            fac = self.db.get_faculty(employee_id)
            name = fac["name"] if fac and fac.get("name") else f"Faculty ({employee_id})"
        except Exception:
            name = f"Faculty ({employee_id})"
        self._name_cache[employee_id] = name
        return name
