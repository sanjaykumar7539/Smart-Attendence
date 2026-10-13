"""
hlc_room_state.py
──────────────────
RoomStateManager — the brain of multi-door HLC tracking.

One state machine per person_id, shared across ALL cameras.

State:
    OUTSIDE  →  (any camera: confirmed IN crossing)  →  INSIDE
    INSIDE   →  (any camera: confirmed OUT crossing)  →  OUTSIDE

Why this is the right model:
    With 2 doors, a person can enter Door A and exit Door B.
    No single camera can track that full journey.
    But the ROOM doesn't care which door was used — it only cares
    whether the person is currently inside or outside.

    The RoomStateManager receives crossing events from ANY camera
    and maintains the single authoritative "is this person inside?" answer.

Anti-double-count logic:
    If Camera A and Camera B BOTH see the same person crossing Door A
    at the same time (due to camera overlap), we only count one event.
    Deduplication window: 5 seconds per person per direction.

Thread safety:
    All state mutations under a single RLock.
    Safe to call from multiple camera reader threads simultaneously.
"""

import threading
import logging
from datetime import datetime, date
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class PersonState:
    """State for a single person."""

    OUTSIDE = "OUTSIDE"
    INSIDE  = "INSIDE"

    def __init__(self, person_id: str):
        self.person_id  = person_id
        self.state      = self.OUTSIDE
        self.session_id: Optional[int] = None   # current open visitor session id

        # Anti-double-count: last time we processed an IN or OUT for this person
        self._last_event_time: Dict[str, datetime] = {}
        self._last_event_cam:  Dict[str, str] = {}   # direction → last camera_id

    def is_inside(self) -> bool:
        return self.state == self.INSIDE

    def can_process(self, direction: str, camera_id: str,
                     dedup_secs: float = 5.0) -> bool:
        """
        Returns True if this event should be processed.
        Suppresses duplicate events from multiple cameras in the same time window.
        """
        now = datetime.now()
        last = self._last_event_time.get(direction)
        if last:
            elapsed = (now - last).total_seconds()
            if elapsed < dedup_secs:
                # Same direction seen recently — is it the same camera or different?
                last_cam = self._last_event_cam.get(direction, "")
                if last_cam != camera_id:
                    # Different camera, same direction, within dedup window
                    # → This is a duplicate cross-detection, suppress it
                    logger.debug(
                        f"DEDUP | {self.person_id} | {direction} from {camera_id} "
                        f"suppressed ({elapsed:.1f}s after {last_cam})"
                    )
                    return False
        return True

    def record_event(self, direction: str, camera_id: str):
        self._last_event_time[direction] = datetime.now()
        self._last_event_cam[direction]  = camera_id


class RoomStateManager:
    """
    Central state manager for HLC room presence.

    Receives crossing events from all cameras.
    Maintains OUTSIDE/INSIDE state per person.
    Returns what action (if any) should be taken: open_session / close_session.
    """

    DEDUP_SECS = 5.0   # suppress duplicate events within this window

    def __init__(self):
        self._states: Dict[str, PersonState] = {}
        self._lock = threading.RLock()

    def _get_or_create(self, person_id: str) -> PersonState:
        if person_id not in self._states:
            self._states[person_id] = PersonState(person_id)
        return self._states[person_id]

    def process_crossing(self,
                          person_id:  str,
                          room_direction: str,    # "IN" or "OUT" (already resolved)
                          camera_id:  str,
                          timestamp:  str) -> Optional[Dict]:
        """
        Main entry point. Call this when a camera confirms a crossing event.

        room_direction: "IN" or "OUT"  (already converted from frame direction
                        by HLCCameraRegistry.resolve_room_direction())

        Returns a dict describing what happened, or None if event was suppressed.

        Return dict keys:
            action      : "session_open" | "session_close" | "reentry_close_then_open"
            person_id   : str
            room_dir    : "IN" | "OUT"
            camera_id   : str
            timestamp   : str
            was_inside  : bool  (state before this event)
            now_inside  : bool  (state after this event)
        """
        with self._lock:
            ps = self._get_or_create(person_id)

            # Anti-double-count
            if not ps.can_process(room_direction, camera_id, self.DEDUP_SECS):
                return None

            ps.record_event(room_direction, camera_id)

            was_inside = ps.is_inside()

            if room_direction == "IN":
                if not was_inside:
                    # Clean entry: OUTSIDE → INSIDE
                    ps.state = PersonState.INSIDE
                    action   = "session_open"
                    logger.info(
                        f"ROOM IN  | {person_id} | cam={camera_id} | {timestamp}"
                    )
                else:
                    # Already inside — re-detection, no state change needed
                    # (e.g. person walked past door A while still inside)
                    logger.debug(
                        f"ROOM IN  (already inside) | {person_id} | "
                        f"cam={camera_id}"
                    )
                    return None

            elif room_direction == "OUT":
                if was_inside:
                    # Clean exit: INSIDE → OUTSIDE
                    ps.state = PersonState.OUTSIDE
                    action   = "session_close"
                    logger.info(
                        f"ROOM OUT | {person_id} | cam={camera_id} | {timestamp}"
                    )
                else:
                    # Already outside — spurious OUT (person passing by corridor?)
                    logger.debug(
                        f"ROOM OUT (already outside) | {person_id} | "
                        f"cam={camera_id}"
                    )
                    return None
            else:
                return None

            return {
                "action":     action,
                "person_id":  person_id,
                "room_dir":   room_direction,
                "camera_id":  camera_id,
                "timestamp":  timestamp,
                "was_inside": was_inside,
                "now_inside": ps.is_inside(),
            }

    def get_session_id(self, person_id: str) -> Optional[int]:
        with self._lock:
            ps = self._states.get(person_id)
            return ps.session_id if ps else None

    def set_session_id(self, person_id: str, session_id: Optional[int]):
        with self._lock:
            ps = self._get_or_create(person_id)
            ps.session_id = session_id

    def is_inside(self, person_id: str) -> bool:
        with self._lock:
            ps = self._states.get(person_id)
            return ps.is_inside() if ps else False

    def get_all_inside(self) -> Dict[str, Optional[int]]:
        """Returns {person_id: session_id} for everyone currently inside."""
        with self._lock:
            return {
                pid: ps.session_id
                for pid, ps in self._states.items()
                if ps.is_inside()
            }

    def force_mark_outside(self, person_id: str, reason: str = "timeout"):
        """Used by SessionTimeoutSweep to clean up stale state."""
        with self._lock:
            ps = self._states.get(person_id)
            if ps and ps.is_inside():
                ps.state      = PersonState.OUTSIDE
                ps.session_id = None
                logger.info(
                    f"FORCE OUTSIDE | {person_id} | reason={reason}"
                )

    def reset_person(self, person_id: str):
        with self._lock:
            self._states.pop(person_id, None)

    def stats(self) -> Dict:
        with self._lock:
            total   = len(self._states)
            inside  = sum(1 for ps in self._states.values() if ps.is_inside())
            outside = total - inside
            return {
                "total_tracked": total,
                "inside":  inside,
                "outside": outside,
            }
