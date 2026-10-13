"""
hlc_processor_v4.py
────────────────────
Multi-door HLC processor. Replaces all previous versions.

Flow per detection:
  1. detection arrives with camera_id
  2. is this an HLC camera?  → registry.is_hlc_camera(camera_id)
  3. resolve room direction   → registry.resolve_room_direction(camera_id, frame_dir)
  4. pass to RoomStateManager → state_mgr.process_crossing(...)
  5. action = "session_open" → open visitor or record team entry
     action = "session_close" → close visitor session or record team exit

HLC team vs visitor distinction happens AFTER room direction is resolved,
so the same logic handles both doors identically.
"""

import logging
from datetime import datetime, date
from typing import Dict, Any, Optional

from hlc_camera_config import HLCCameraRegistry
from hlc_room_state    import RoomStateManager
from hlc_team_db       import HLCTeamDB
from hlc_visitor_db_v4 import HLCVisitorDBV4 as HLCVisitorDB

logger = logging.getLogger(__name__)


class HLCProcessorV4:

    def __init__(self,
                 camera_registry: HLCCameraRegistry,
                 room_state:      RoomStateManager,
                 team_db:         HLCTeamDB,
                 visitor_db:      HLCVisitorDB):
        self.registry    = camera_registry
        self.room        = room_state
        self.tdb         = team_db
        self.vdb         = visitor_db

    # ── Entry point ───────────────────────────────────────────────────────────

    def process_detection(self, detection: Dict[str, Any]) -> Optional[Dict]:
        """
        Call for every face detection from any HLC camera.

        Required keys in detection dict:
            employee_id     str
            name            str
            camera_id       str       — must match a registered HLC camera
            direction       str       — tracker's raw direction (frame-relative)
                                        e.g. "top_to_bottom", "bottom_to_top",
                                             "IN", "OUT" (legacy)
            time            str       — HH:MM:SS
            confidence_raw  float     — 0.0–1.0
            emp_info        dict|None — dept etc. from employees table
        """
        camera_id   = detection.get("camera_id", "")
        employee_id = detection.get("employee_id")
        name        = detection.get("name", "Unknown")
        raw_dir     = detection.get("direction", "")
        time_str    = detection.get("time",
                       datetime.now().strftime("%H:%M:%S"))
        confidence  = float(detection.get("confidence_raw", 0))
        today       = date.today().isoformat()

        # ── 1. Is this even an HLC camera? ────────────────────────────────────
        if not self.registry.is_hlc_camera(camera_id):
            return None   # not our camera — let main attendance handle it

        # ── 2. Unknown face ───────────────────────────────────────────────────
        if not employee_id or employee_id == "Unknown":
            cam = self.registry.get(camera_id)
            logger.warning(
                f"HLC UNKNOWN | cam={camera_id} "
                f"door={cam.door_id if cam else '?'} | {time_str}"
            )
            return {
                "type":      "hlc_unknown",
                "camera_id": camera_id,
                "time":      time_str,
                "flagged":   True,
                "alert":     True,
            }

        # ── 3. Resolve frame direction → room direction ───────────────────────
        room_dir = self.registry.resolve_room_direction(camera_id, raw_dir)
        if not room_dir:
            logger.debug(
                f"HLC direction unresolved | {employee_id} | "
                f"cam={camera_id} raw={raw_dir}"
            )
            return None

        # ── 4. Run through RoomStateManager ──────────────────────────────────
        crossing = self.room.process_crossing(
            person_id     = employee_id,
            room_direction= room_dir,
            camera_id     = camera_id,
            timestamp     = time_str,
        )

        if crossing is None:
            # Suppressed (duplicate or no state change)
            return None

        action = crossing["action"]

        # ── 5. Route to team or visitor handler ───────────────────────────────
        emp             = detection.get("emp_info") or {}
        dept            = emp.get("dept", "")
        designation     = emp.get("designation", "")
        staff_category  = emp.get("staff_category", "Teaching")   # Teaching | Non-Teaching

        # faculty.is_hlc (master DB) is the source of truth. hlc_team.db is
        # checked as a fallback only for anyone not yet migrated there —
        # keeps old enrollments working without a forced data migration.
        is_hlc_member = bool(emp.get("is_hlc")) or self.tdb.is_hlc_employee(employee_id)

        if is_hlc_member:
            return self._handle_team(
                employee_id, name, today, time_str,
                room_dir, action, camera_id
            )
        else:
            return self._handle_visitor(
                employee_id, name, dept, designation, staff_category,
                today, time_str, room_dir, action, camera_id, confidence
            )

    # ── HLC Team ──────────────────────────────────────────────────────────────

    def _handle_team(self, employee_id, name, today, time_str,
                      room_dir, action, camera_id) -> Dict:
        cam = self.registry.get(camera_id)
        door = cam.door_id if cam else "?"

        if action == "session_open":    # entered room
            result = self.tdb.record_entry(employee_id, today, time_str)
            logger.info(
                f"HLC TEAM ENTRY | {employee_id} {name} | "
                f"door={door} cam={camera_id} | {time_str}"
            )
        else:                           # left room
            self.tdb.record_exit(employee_id, today, time_str)
            logger.info(
                f"HLC TEAM EXIT  | {employee_id} {name} | "
                f"door={door} cam={camera_id} | {time_str}"
            )

        return {
            "type":        "hlc_team",
            "employee_id": employee_id,
            "name":        name,
            "room_dir":    room_dir,
            "door_id":     door,
            "camera_id":   camera_id,
            "time":        time_str,
            "action":      action,
            "flagged":     False,
        }

    # ── Visitor ────────────────────────────────────────────────────────────────

    def _handle_visitor(self, visitor_id, name, dept, designation,
                          staff_category, today, time_str, room_dir,
                          action, camera_id, confidence) -> Dict:
        cam   = self.registry.get(camera_id)
        door  = cam.door_id if cam else "?"

        if action == "session_open":
            # ── New entry ──────────────────────────────────────────────────
            # person_type now reflects the real staff_category (Teaching /
            # Non-Teaching) from the master faculty table instead of being
            # hardcoded to "FACULTY" — this is what lets alerts/reports show
            # "Associate Professor" vs "Admin Executive" for non-HLC visitors.
            sid = self.vdb.open_session(
                visitor_id  = visitor_id,
                name        = name,
                dept        = dept,
                person_type = staff_category,
                designation = designation,
                visit_date  = today,
                entry_time  = time_str,
                confidence  = confidence,
                camera_id   = camera_id,
                door_id     = door,
            )
            # Store session_id in room state for quick exit lookup
            self.room.set_session_id(visitor_id, sid)

            sess      = self.vdb.get_open_session(visitor_id, today)
            visit_num = sess["visit_number"] if sess else 1

            logger.info(
                f"HLC VISITOR ENTRY | {visitor_id} {name} | "
                f"visit #{visit_num} | door={door} cam={camera_id} | {time_str}"
            )
            return {
                "type":         "hlc_visitor",
                "sub_type":     "entry",
                "visitor_id":   visitor_id,
                "name":         name,
                "dept":         dept,
                "designation":  designation,
                "staff_category": staff_category,
                "room_dir":     "IN",
                "door_id":      door,
                "camera_id":    camera_id,
                "time":         time_str,
                "session_id":   sid,
                "visit_number": visit_num,
                "flagged":      True,
                "alert":        True,
            }

        else:   # session_close
            # ── Exit ──────────────────────────────────────────────────────────
            # Get open session — could have entered via the OTHER door
            open_sess = self.vdb.get_open_session(visitor_id, today)
            if not open_sess:
                logger.debug(
                    f"HLC VISITOR OUT with no open session | "
                    f"{visitor_id} | door={door}"
                )
                self.room.set_session_id(visitor_id, None)
                return {
                    "type":     "hlc_visitor",
                    "sub_type": "exit_no_session",
                    "visitor_id": visitor_id,
                    "room_dir": "OUT",
                    "door_id":  door,
                    "time":     time_str,
                }

            result = self.vdb.close_session_with_method(
                session_id  = open_sess["id"],
                exit_time   = time_str,
                exit_method = "CAMERA",
                exit_door   = door,       # record which door they left through
            )
            self.room.set_session_id(visitor_id, None)

            logger.info(
                f"HLC VISITOR EXIT | {visitor_id} {name} | "
                f"entry_door={open_sess.get('door_id','?')} "
                f"exit_door={door} | "
                f"duration={result.get('duration_min')}m | cam={camera_id}"
            )
            return {
                "type":         "hlc_visitor",
                "sub_type":     "exit",
                "visitor_id":   visitor_id,
                "name":         name,
                "room_dir":     "OUT",
                "door_id":      door,
                "entry_door":   open_sess.get("door_id", "?"),
                "camera_id":    camera_id,
                "time":         time_str,
                "duration_min": result.get("duration_min"),
                "over_15min":   result.get("over_15min", False),
                "exit_method":  "CAMERA",
                "flagged":      True,
            }
