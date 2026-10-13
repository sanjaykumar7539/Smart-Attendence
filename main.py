"""
FaceTrack Pro — main.py
Camera loop + dispatcher + FastAPI web server.

Architecture:
  main.py (thin dispatcher)
    ├── CAM-01 (main camera)  → attendance.py         → attendance.db
    ├── HLC-CAM-*             → hlc_processor_v4.py   → hlc_team.db / hlc_visitors.db
    └── (future cameras)      → their own processor

Bug fixes applied:
  FIX-L1  CameraReader: GStreamer low-latency
  FIX-L2  /stream: asyncio.sleep(0)
  FIX-L3  Frame-skip flicker: cache last_annotated
  FIX-L4  Direction: prev_cy updated after not before
  FIX-L5  Recognition: sticky (never reverts to Unknown)
  FIX-DB2 asyncio: run_coroutine_threadsafe not asyncio.run()
"""

if __name__ == "__main__":
    from run_integrated import main as launch_integrated
    launch_integrated()
    raise SystemExit

import cv2
import numpy as np
import asyncio
import threading
import time
import os
import json
from datetime import date, datetime
from typing import Dict, Set, Optional
from pathlib import Path
from loguru import logger

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv()   # must run before `from config import config` below —
                 # config.py reads GEMINI_API_KEY via os.getenv() at import time

# File logging with automatic rotation — keeps console output too (loguru's
# default sink). Without this, "python main.py" only logs to the console;
# closing the terminal loses everything, and nothing rotates/cleans up.
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/facetrack_{time:YYYY-MM-DD}.log",
    rotation="1 day",       # new file each day
    retention="14 days",    # auto-delete anything older than 2 weeks
    encoding="utf-8",
    enqueue=True,           # thread-safe, since multiple threads log here
    level="INFO",
)

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from typing import Literal
from fastapi.responses import HTMLResponse, Response, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import config
from database import db
from recognition import RecognitionEngine
from adaptive_recognition import AdaptiveRecognitionManager
from tracker import SORTTracker, KalmanFaceTrack, iou as compute_iou
from camera_reader import CameraReader
from hr_routes import router as hr_router
from audit_engine import audit_engine

# ── Main attendance module ────────────────────────────────────────────────────
from attendance import AttendanceProcessor

# ── HLC modules ───────────────────────────────────────────────────────────────
from hlc_camera_config import HLCCameraRegistry, build_registry_from_config
from hlc_room_state    import RoomStateManager
from hlc_processor_v4  import HLCProcessorV4
from hlc_team_db       import HLCTeamDB
from hlc_visitor_db_v4 import HLCVisitorDBV4
from hlc_routes_v2     import hlc_router, init_hlc
from hlc_alert_engine  import HLCAlertEngine
from hlc_exit_handler  import SessionTimeoutSweep

# ── AI module ─────────────────────────────────────────────────────────────────
from ai_routes_v2 import ai_router, init_ai


# ── Shared state ──────────────────────────────────────────────────────────────

class AppState:
    latest_frame:      bytes = None
    hlc_frames:        Dict[str, bytes] = {}   # camera_id -> latest JPEG frame
    latest_events:     list  = []
    active_websockets: Set[WebSocket] = set()
    running:           bool  = True
    fps_actual:        float = 0.0
    inference_fps:     float = 0.0
    faces_in_frame:    int   = 0
    live_counts = {"detected": 0, "confirmed": 0, "unknown": 0}
    attendance_lock = threading.RLock()
    reset_generation = 0
    marked_today: Set[str] = set()
    marked_date: str = datetime.now().date().isoformat()


state        = AppState()
_server_ready = threading.Event()


# ── Processor registry ────────────────────────────────────────────────────────
# Holds all camera → processor mappings.
# main_attendance: AttendanceProcessor  (for CAM-01)
# hlc_proc:        HLCProcessorV4       (for HLC-CAM-*)
# hlc_registry:    HLCCameraRegistry    (knows which cam_ids are HLC)

main_attendance: AttendanceProcessor = None
hlc_proc:        HLCProcessorV4      = None
hlc_registry:    HLCCameraRegistry   = None
hlc_camera_readers: Dict[str, CameraReader] = {}


# ── Attendance trigger (line crossing) ───────────────────────────────────────

class AttendanceTrigger:
    """
    Marks confirmed identities once per day in presence mode, or fires when a
    confirmed face track crosses the virtual detection line in entry mode.

    IMPORTANT: a track crossing the line and a track being RECOGNIZED
    don't necessarily happen on the same frame — a fast-moving person may
    cross the line before recognition has converged on their identity.
    The old version required both to be true simultaneously, which meant
    any crossing that happened before recognition finished was lost
    forever (by the time recognition succeeded, they were already past
    the line — no new crossing transition to detect).
    Now: the crossing itself is remembered the moment it's seen,
    regardless of recognition status, and dispatched as soon as identity
    resolves — even if that's a few frames later.
    """
    def __init__(self):
        self._cooldowns: Dict[int, float] = {}
        self._present_today: Set[str] = set()
        self._presence_date: Optional[date] = None
        self._pending:   Dict[int, tuple] = {}   # track_id -> (direction, first_seen_ts)
        self._pending_timeout_sec = 8   # give up waiting for recognition after this long

    def check(self, track: KalmanFaceTrack, frame_h: int) -> bool:
        if config.tracker.attendance_mode == "presence":
            # Only a currently observed, confirmed identity can mark attendance.
            if not track.recognized or not track.employee_id or track.no_losses:
                return False
            today = datetime.now().date()
            if today != self._presence_date:
                self._present_today.clear()
                self._presence_date = today
            if track.employee_id in self._present_today:
                return False
            self._present_today.add(track.employee_id)
            track.last_direction = "IN"
            return True
        line_y = frame_h * config.tracker.line_position
        cy     = track.get_center_y()

        if track.prev_cy is None:
            track.prev_cy = cy
            return False

        # FIX-L4: read THEN update
        prev_cy       = track.prev_cy
        track.prev_cy = cy
        crossed = (prev_cy < line_y <= cy) or (prev_cy > line_y >= cy)

        # This track already crossed on an earlier frame and is waiting
        # for recognition to catch up — check if it has now.
        if track.track_id in self._pending:
            direction, first_seen = self._pending[track.track_id]

            if track.recognized and track.employee_id is not None:
                self._pending.pop(track.track_id)
                now  = time.time()
                last = self._cooldowns.get(track.track_id, 0)
                if now - last < config.tracker.trigger_cooldown_sec:
                    return False
                self._cooldowns[track.track_id] = now
                track.last_direction = direction
                return True

            # Give up after a bounded wait — recognition genuinely never
            # converged (bad angle, glare, etc). Prevents this dict from
            # growing forever over weeks of continuous operation.
            if time.time() - first_seen > self._pending_timeout_sec:
                self._pending.pop(track.track_id)
            return False

        if not crossed:
            return False

        direction = "IN" if cy > prev_cy else "OUT"

        if not track.recognized or track.employee_id is None:
            # Crossed the line but identity isn't resolved yet — remember
            # it instead of losing it. Will fire above once recognized,
            # as long as the track survives (max_age) that long.
            self._pending[track.track_id] = (direction, time.time())
            return False

        now  = time.time()
        last = self._cooldowns.get(track.track_id, 0)
        if now - last < config.tracker.trigger_cooldown_sec:
            return False
        self._cooldowns[track.track_id] = now
        track.last_direction = "IN" if cy > prev_cy else "OUT"
        return True


class HLCDoorTrigger:
    """
    Simplified entry/exit trigger for the HLC dual-camera door setup —
    no line-crossing geometry needed. With one camera outside the door
    and one inside, the camera that recognized the person IS the
    direction signal:

      - Recognized on the INSIDE camera, and room state says they're not
        currently marked inside -> they just entered.
      - Recognized on the OUTSIDE camera, and room state says they ARE
        currently marked inside -> they just exited.
      - Any other combination -> no event (e.g. seen outside while
        already outside — they haven't gone through the door yet).

    Each track fires at most once per lifetime, so someone lingering in
    view of one camera doesn't spam repeat events.
    """
    def __init__(self):
        self._fired: set = set()

    def check(self, track: KalmanFaceTrack, camera_role: str,
              currently_inside: bool) -> Optional[str]:
        """
        camera_role: "inside" or "outside"
        currently_inside: room_state.is_inside(employee_id) right now
        Returns "IN", "OUT", or None.
        """
        if not track.recognized or track.employee_id is None:
            return None
        if track.track_id in self._fired:
            return None

        if camera_role == "inside" and not currently_inside:
            self._fired.add(track.track_id)
            return "IN"
        if camera_role == "outside" and currently_inside:
            self._fired.add(track.track_id)
            return "OUT"
        return None

    def cleanup(self, active_track_ids: set):
        """Call periodically to drop fired-flags for tracks no longer
        active, so this set doesn't grow forever over long runs."""
        self._fired &= active_track_ids


# ── Processing loop ───────────────────────────────────────────────────────────

def processing_loop(
    engine:       RecognitionEngine,
    camera:       CameraReader,
    broadcast_fn,
    loop:         asyncio.AbstractEventLoop,
    preview=None,
):
    """
    Background thread: reads frames → AI inference → dispatches to correct
    attendance processor based on camera_id.
    """
    tracker     = SORTTracker(
        max_age       = config.tracker.max_age,
        min_hits      = config.tracker.min_hits,
        iou_threshold = config.tracker.iou_threshold,
    )
    trigger     = AttendanceTrigger()
    generation = state.reset_generation
    adaptive    = AdaptiveRecognitionManager(engine, config.model) if config.model.adaptive_recognition else None
    frame_idx   = 0
    fps_timer   = time.time()
    fps_counter = 0
    name_cache: Dict[str, str] = {}

    # FIX-L3: cache last annotated frame
    last_annotated = None

    # Startup blank frame
    blank = np.zeros(
        (config.camera.frame_height, config.camera.frame_width, 3), np.uint8
    )
    _, j = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 60])
    state.latest_frame = j.tobytes()

    logger.info("Processing loop: waiting for server...")
    _server_ready.wait()
    logger.success("Processing loop: running")

    while state.running:
        frame = camera.get_new_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_idx += 1
        h, w = frame.shape[:2]

        run_inference = (frame_idx % config.camera.frame_skip == 0)

        if run_inference:
            inference_started = time.monotonic()
            if generation != state.reset_generation:
                generation = state.reset_generation
                tracker.tracks.clear()
                if config.model.detector == "AdaFace original":
                    engine.reset()
                trigger = AttendanceTrigger()
                if adaptive is not None:
                    adaptive.cleanup([])
            try:
                faces_data, _ = engine.process_frame(frame, annotate=False)
                # Render only confirmed track labels, not a second set of single-frame guesses.
                annotated = frame.copy()
            except Exception as e:
                logger.error(f"process_frame error: {e}")
                faces_data = []
                annotated  = frame.copy()

            detections    = [
                {"bbox": f["bbox"], "confidence": f["confidence"],
                 "embedding": f.get("embedding")}
                for f in faces_data
            ]
            active_tracks = tracker.update(detections)
            if adaptive is not None:
                adaptive.cleanup([t.track_id for t in tracker.tracks])

            for track in active_tracks:
                # Track-aware adaptive recognition. Identity becomes sticky ONLY after
                # quality-adaptive search + temporal voting reaches CONFIRMED.
                if not track.recognized:
                    best_iou, best_face = 0.0, None
                    for face in faces_data:
                        score = compute_iou(np.array(track.get_bbox()), np.array(face["bbox"]))
                        if score > best_iou:
                            best_iou, best_face = score, face

                    if best_face and best_iou > 0.3:
                        if adaptive is not None:
                            decision = adaptive.update(track.track_id, frame, best_face)
                            track.recognition_status = decision.status
                            track.quality_level = decision.quality
                            track.was_enhanced = decision.enhanced
                            if decision.status == "CONFIRMED" and decision.employee_id:
                                track.employee_id = decision.employee_id
                                track.confidence = decision.confidence
                                track.recognized = True
                        elif best_face["recognized"]:
                            track.employee_id = best_face["employee_id"]
                            track.confidence = best_face["similarity"]
                            track.recognized = True
                            track.recognition_status = "CONFIRMED"

                        if track.recognized:
                            eid = track.employee_id
                            if eid not in name_cache:
                                fac = db.get_faculty(eid)
                                name_cache[eid] = (fac["name"] if fac and fac.get("name") else f"Faculty ({eid})")
                            track.name = name_cache[eid]

                # ── Line crossing check ───────────────────────────────────────
                if not trigger.check(track, h):
                    continue

                direction = track.last_direction
                cam_id    = config.camera.camera_id   # "CAM-01"

                detection_dict = {
                    "employee_id":    track.employee_id,
                    "name":           track.name,
                    "direction":      direction,
                    "confidence":     track.confidence,
                    "confidence_raw": track.confidence,
                    "track_id":       track.track_id,
                    "time":           datetime.now().strftime("%H:%M:%S"),
                    "camera_id":      cam_id,
                    "generation":     generation,
                }

                # ══════════════════════════════════════════════════════════════
                # DISPATCHER — routes to the correct processor by camera_id
                # ══════════════════════════════════════════════════════════════
                result = _dispatch(detection_dict)
                # ══════════════════════════════════════════════════════════════

                if result:
                    employee_id = result.get("employee_id")
                    if employee_id:
                        with state.attendance_lock:
                            today = datetime.now().date().isoformat()
                            if state.marked_date != today:
                                state.marked_today.clear()
                                state.marked_date = today
                            state.marked_today.add(employee_id)
                    # Determine WS event type
                    if result.get("type") == "hlc_event" or result.get("type", "").startswith("hlc"):
                        ws_payload = {"type": "hlc_event", "payload": result}
                    else:
                        # Main attendance event
                        state.latest_events.insert(0, result)
                        state.latest_events = state.latest_events[:50]
                        logger.success(
                            f"ATTENDANCE ▶ {result.get('name')} | "
                            f"{direction} | {result.get('status')}"
                        )
                        ws_payload = {"type": "attendance", "data": result}

                    # FIX-DB2: thread-safe WS broadcast
                    asyncio.run_coroutine_threadsafe(
                        broadcast_fn(ws_payload), loop
                    )

            # Include newly detected, unconfirmed tracks, not only min_hits tracks.
            visible = [t for t in tracker.tracks if t.no_losses <= 8]
            people = getattr(engine, 'last_person_boxes', [])
            unmatched_people = _unmatched_people(visible, people)
            confirmed = sum(_confirmed(t) for t in visible)
            total = len(visible) + len(unmatched_people)
            state.live_counts = {"detected": total, "confirmed": confirmed, "unknown": total - confirmed}
            state.faces_in_frame = len(visible)
            state.inference_fps = 1.0 / max(time.monotonic() - inference_started, 0.001)
            if preview is not None:
                preview.update_overlays(visible, unmatched_people, generation,
                                        source_frame=frame, source_time=inference_started)
                continue
            _draw_overlays(annotated, visible, h, w, unmatched_people)
            last_annotated = annotated.copy()

        else:
            if preview is not None:
                continue
            # FIX-L3: reuse last annotated on skipped frames
            if last_annotated is not None:
                annotated = last_annotated.copy()
                _draw_hud(annotated, w, h)
            else:
                annotated = frame.copy()
                _draw_hud(annotated, w, h)

        # FIX-L2: encode frame
        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
        state.latest_frame = jpeg.tobytes()

        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            state.fps_actual = fps_counter
            fps_counter = 0
            fps_timer   = time.time()


def hlc_processing_loop(
    camera_id:    str,
    camera_role:  str,     # "inside" or "outside"
    camera:       CameraReader,
    engine:       RecognitionEngine,
    broadcast_fn,
    loop:         asyncio.AbstractEventLoop,
):
    """
    Same shape as processing_loop() above, generalized for one HLC camera.
    Runs recognition on this camera's own frames only, tags every
    detection with THIS camera's id (so _dispatch() routes it to
    hlc_proc, not the main attendance processor), and writes annotated
    frames into state.hlc_frames[camera_id] instead of the shared
    state.latest_frame the main camera uses.

    Direction comes from HLCDoorTrigger (which camera + current room
    state) — no line-crossing geometry, unlike the main camera's
    AttendanceTrigger.

    One instance of this runs per HLC camera (see main() below) — each
    with its OWN SORTTracker/HLCDoorTrigger so track IDs never bleed
    between cameras.
    """
    tracker = SORTTracker(
        max_age       = config.tracker.max_age,
        min_hits      = config.tracker.min_hits,
        iou_threshold = config.tracker.iou_threshold,
    )
    trigger        = HLCDoorTrigger()
    frame_idx      = 0
    name_cache: Dict[str, str] = {}
    last_annotated = None

    blank = np.zeros(
        (config.camera.frame_height, config.camera.frame_width, 3), np.uint8
    )
    _, j = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 60])
    state.hlc_frames[camera_id] = j.tobytes()

    _server_ready.wait()
    logger.success(f"HLC processing loop [{camera_id}]: running")

    while state.running:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_idx += 1
        h, w = frame.shape[:2]
        run_inference = (frame_idx % config.camera.frame_skip == 0)

        if run_inference:
            try:
                faces_data, annotated = engine.process_frame(frame)
            except Exception as e:
                logger.error(f"HLC[{camera_id}] process_frame error: {e}")
                faces_data = []
                annotated  = frame.copy()

            detections    = [
                {"bbox": f["bbox"], "confidence": f["confidence"]}
                for f in faces_data
            ]
            active_tracks = tracker.update(detections)
            if adaptive is not None:
                adaptive.cleanup([t.track_id for t in active_tracks])

            for track in active_tracks:
                if not track.recognized:
                    best_iou, best_face = 0.0, None
                    for face in faces_data:
                        score = compute_iou(
                            np.array(track.get_bbox()),
                            np.array(face["bbox"])
                        )
                        if score > best_iou:
                            best_iou, best_face = score, face

                    if best_face and best_face["recognized"] and best_iou > 0.3:
                        track.employee_id = best_face["employee_id"]
                        track.confidence  = best_face["similarity"]
                        track.recognized  = True
                        eid = track.employee_id
                        if eid not in name_cache:
                            fac = db.get_faculty(eid)
                            name_cache[eid] = (
                                fac["name"] if fac and fac.get("name")
                                else f"Faculty ({eid})"
                            )
                        track.name = name_cache[eid]

                if not track.recognized or track.employee_id is None:
                    continue

                currently_inside = (
                    hlc_proc.room.is_inside(track.employee_id)
                    if hlc_proc is not None else False
                )
                direction = trigger.check(track, camera_role, currently_inside)
                if direction is None:
                    continue

                # Pull real role data (designation, staff_category, dept, is_hlc)
                # from the master faculty table so the HLC processor can tell
                # a Non-Teaching admin from an Associate Professor instead of
                # defaulting everyone to "FACULTY". Cached per-track_id loop
                # pass via name_cache lookup above already happened; fetch
                # fresh here since staff_category/designation aren't cached.
                fac_info = db.get_faculty(track.employee_id) or {}

                detection_dict = {
                    "employee_id":    track.employee_id,
                    "name":           track.name,
                    "direction":      direction,
                    "confidence":     track.confidence,
                    "confidence_raw": track.confidence,
                    "track_id":       track.track_id,
                    "time":           datetime.now().strftime("%H:%M:%S"),
                    "camera_id":      camera_id,
                    "emp_info": {
                        "dept":           fac_info.get("department", ""),
                        "designation":    fac_info.get("designation", ""),
                        "staff_category": fac_info.get("staff_category", "Teaching"),
                        "is_hlc":         bool(fac_info.get("is_hlc", 0)),
                    },
                }

                result = _dispatch(detection_dict)

                if result:
                    ws_payload = {"type": "hlc_event", "payload": result}
                    asyncio.run_coroutine_threadsafe(broadcast_fn(ws_payload), loop)

            trigger.cleanup({t.track_id for t in active_tracks})
            _draw_overlays(annotated, active_tracks, h, w)
            last_annotated = annotated.copy()
        else:
            if last_annotated is not None:
                annotated = last_annotated.copy()
                _draw_hud(annotated, w, h)
            else:
                annotated = frame.copy()
                _draw_hud(annotated, w, h)

        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
        state.hlc_frames[camera_id] = jpeg.tobytes()


def _dispatch(detection: Dict) -> Dict:
    with state.attendance_lock:
        if detection.get('generation', state.reset_generation) != state.reset_generation:
            return None
        result = _dispatch_unlocked(detection)
        if result:
            result['work_date'] = datetime.now().date().isoformat()
            result['generation'] = state.reset_generation
        return result


def _dispatch_unlocked(detection: Dict) -> Dict:
    """
    Route detection to the correct processor based on camera_id.

    CAM-01 (staff room)  → attendance.py   (main_attendance)
    HLC-CAM-*            → hlc_processor_v4.py (hlc_proc)

    Adding a new location later:
        elif cam_id == "CANTEEN-CAM-01":
            return canteen_proc.process_detection(detection)
    """
    cam_id = detection.get("camera_id", "")

    # ── HLC cameras ───────────────────────────────────────────────────────────
    if (hlc_registry is not None and
            hlc_registry.is_hlc_camera(cam_id) and
            hlc_proc is not None):
        return hlc_proc.process_detection(detection)

    # ── Main staff-room camera (default) ──────────────────────────────────────
    if main_attendance is not None:
        return main_attendance.process_detection(detection)

    return None


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _get_status(employee_id: str) -> str:
    for r in db.get_today_attendance():
        if r["employee_id"] == employee_id:
            return r["final_status"]
    return "LOGGED"


def _confirmed(track):
    return bool(track.recognized and track.employee_id and
                track.recognition_status == 'CONFIRMED')


def _attendance_marked(track):
    with state.attendance_lock:
        return bool(track.employee_id and track.employee_id in state.marked_today)


def _unmatched_people(tracks, people):
    """A face can replace only one person box, even in an overlapping classroom row."""
    matched = set()
    for track in tracks:
        fx1, fy1, fx2, fy2 = track.get_bbox()
        cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
        candidates = [i for i, (x1, y1, x2, y2) in enumerate(people)
                      if i not in matched and x1 <= cx <= x2 and y1 <= cy <= y2]
        if candidates:
            matched.add(min(candidates, key=lambda i:
                (people[i][2] - people[i][0]) * (people[i][3] - people[i][1])))
    return [box for i, box in enumerate(people) if i not in matched]


def _draw_overlays(frame: np.ndarray, tracks, h: int, w: int, unmatched_people=()):
    cfg    = config.display
    if config.tracker.attendance_mode == "line":
        line_y = int(h * config.tracker.line_position)
        for x in range(0, w, 60):
            cv2.line(frame, (x, line_y), (min(x + 30, w), line_y), (255, 200, 0), 2)
        cv2.putText(
            frame, "DETECTION LINE", (10, line_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA,
        )
    for track in tracks:
        bbox        = np.clip(track.get_bbox().astype(int), [0, 0, 0, 0], [w - 1, h - 1, w - 1, h - 1])
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue
        known       = _confirmed(track) or _attendance_marked(track)
        color       = (0, 255, 0) if known else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        if known:
            cv2.putText(
                frame, track.name, (x1, y1 - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 1, cv2.LINE_AA,
            )
            cv2.putText(
                frame, f"{track.confidence:.0%}", (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA,
            )
    for box in unmatched_people:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
    _draw_hud(frame, w, h)


def _draw_hud(frame: np.ndarray, w: int, h: int):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 46), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    now = datetime.now().strftime("%d %b %Y  %H:%M:%S")
    cv2.putText(
        frame, config.camera.location.upper(), (12, 29),
        cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 212, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, now, (w - 290, 29),
        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (170, 170, 170), 1, cv2.LINE_AA,
    )
    if config.tracker.attendance_mode != "presence":
        cv2.putText(
            frame, f"Total:{state.live_counts['detected']}  Confirmed:{state.live_counts['confirmed']}  Unknown:{state.live_counts['unknown']}",
            (w // 2 - 210, 29),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA,
        )
    if int(time.time()) % 2 == 0:
        cv2.circle(frame, (w - 18, 22), 6, (0, 0, 220), -1)


# ── FastAPI application ───────────────────────────────────────────────────────

app = FastAPI(title="FaceTrack Pro", version="2.0.0")

app.include_router(hr_router)
app.include_router(hlc_router)          # HLC routes  (/api/hlc/*)
app.include_router(ai_router, prefix="/api/ai")  # AI routes   (/api/ai/*)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def on_startup():
    _server_ready.set()
    logger.success("FastAPI started — processing loop unblocked")
    if config.ai.enabled:
        init_ai(
            db_path        = config.database.db_path,
            hlc_team_db    = config.hlc.team_db_path,
            hlc_visitor_db = config.hlc.visitor_db_path,
            api_key        = config.ai.api_key,
        )
        logger.info("AI module ready via FastAPI startup")


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    state.active_websockets.add(ws)
    try:
        await ws.send_json({
            "type":   "init",
            "stats":  _dashboard_stats(),
            "events": _today_events()[:20],
        })
        while True:
            await asyncio.sleep(2)
            await ws.send_json({
                "type":  "stats",
                "stats": _dashboard_stats(),
                "fps":   state.fps_actual,
            })
    except WebSocketDisconnect:
        state.active_websockets.discard(ws)
    except Exception:
        state.active_websockets.discard(ws)


async def broadcast(payload: dict):
    dead = set()
    for ws in state.active_websockets:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.add(ws)
    state.active_websockets -= dead


# ── MJPEG stream ──────────────────────────────────────────────────────────────

@app.get("/stream")
async def video_stream():
    """Main camera MJPEG stream. FIX-L2: asyncio.sleep(0)"""
    async def generate():
        last_sent = None
        while state.running:
            frame = state.latest_frame
            if frame is not None and frame is not last_sent:
                last_sent = frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                    + frame
                    + b"\r\n"
                )
            await asyncio.sleep(0.01)
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/api/hlc/cameras")
async def hlc_cameras_list():
    """List registered HLC cameras — used by the dashboard to render one
    video tile per camera instead of a hardcoded single feed."""
    if hlc_registry is None:
        return {"cameras": []}
    return {
        "cameras": [
            {"camera_id": c.camera_id, "door_id": c.door_id, "label": c.label}
            for c in hlc_registry.all_enabled()
        ]
    }


@app.get("/stream/hlc")
async def hlc_stream_default():
    """MJPEG stream from the first enabled HLC camera."""
    if hlc_registry is None:
        return JSONResponse({"error": "HLC not initialised"}, 503)
    cams = hlc_registry.all_enabled()
    if not cams:
        return JSONResponse({"error": "No HLC cameras configured"}, 503)
    return await hlc_stream_by_id(cams[0].camera_id)


@app.get("/stream/hlc/{camera_id}")
async def hlc_stream_by_id(camera_id: str):
    """MJPEG stream for one specific HLC camera."""
    if hlc_registry is None:
        return JSONResponse({"error": "HLC not initialised"}, 503)
    cam = hlc_registry.get(camera_id)
    if not cam:
        return JSONResponse({"error": f"Camera {camera_id} not found"}, 404)

    async def generate():
        last_sent = None
        while state.running:
            frame = state.hlc_frames.get(camera_id)
            if frame is not None and frame is not last_sent:
                last_sent = frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            await asyncio.sleep(0)
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── REST API ──────────────────────────────────────────────────────────────────

@app.get("/api/college-events")
async def college_events():
    """
    Reads ./college_events.json — a simple editable list of upcoming college
    events/holidays shown in the dashboard widget. Returns [] if the file
    doesn't exist yet (widget just shows nothing, no error).
    """
    path = "college_events.json"
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/stats")
async def get_stats():
    return _dashboard_stats()


def _dashboard_stats():
    if config.tracker.attendance_mode == "presence":
        roster = _dashboard_roster()
        records = {
            row["employee_id"]: row["final_status"]
            for row in db.get_today_attendance()
        }
        present = sum(records.get(person["employee_id"]) == "PRESENT" for person in roster)
        return {
            "total": len(roster),
            "present": present,
            "permission": 0,
            "half_day": 0,
            "absent": len(roster) - present,
            "date": datetime.now().date().isoformat(),
            "live": state.live_counts,
            "fps": state.fps_actual,
            "generation": state.reset_generation,
        }
    return {**db.get_today_stats(), 'live': state.live_counts,
            'fps': state.fps_actual, 'generation': state.reset_generation}


def _dashboard_roster():
    faculty = db.get_all_faculty()
    roster_ids = config.database.student_roster_ids
    if config.tracker.attendance_mode != "presence" or roster_ids is None:
        return faculty
    faculty_by_id = {person["employee_id"]: person for person in faculty}
    unique_roster_ids = list(dict.fromkeys(roster_ids))
    missing_ids = [identity for identity in unique_roster_ids if identity not in faculty_by_id]
    if missing_ids:
        logger.error(f"Student roster is missing database records: {missing_ids}")
        raise HTTPException(500, "Student roster is not fully initialized")
    return [faculty_by_id[identity] for identity in unique_roster_ids]


def _today_events():
    today = datetime.now().date().isoformat()
    events = [e for e in state.latest_events if e.get('work_date') == today and
              e.get('generation') == state.reset_generation]
    if config.tracker.attendance_mode == "presence" and config.database.student_roster_ids is not None:
        roster_ids = set(config.database.student_roster_ids)
        events = [event for event in events if event.get("employee_id") in roster_ids]
    return events


@app.get("/api/attendance")
async def get_attendance():
    today = datetime.now().date().isoformat()
    records = {r['employee_id']: r for r in db.get_today_attendance()}
    return [{**person, 'work_date': today, 'final_status': 'ABSENT',
             'morning_status': 'ABSENT', 'evening_status': 'ABSENT',
             'first_seen': None, 'last_seen': None,
             **records.get(person['employee_id'], {}), 'student_id': person['employee_id']}
            for person in _dashboard_roster()]


@app.get("/api/events")
async def get_events():
    return _today_events()


@app.post('/api/attendance/reset')
async def reset_attendance(scope: Literal['today', 'previous']):
    if config.tracker.attendance_mode != 'presence':
        raise HTTPException(400, 'Reset is available only in student presence mode')
    with state.attendance_lock:
        result = db.reset_presence(scope)
        if scope == 'today':
            state.reset_generation += 1
            state.latest_events = []
            state.marked_today.clear()
            state.marked_date = datetime.now().date().isoformat()
            state.live_counts = {'detected': 0, 'confirmed': 0, 'unknown': 0}
        stats = _dashboard_stats()
    await broadcast({'type': 'attendance_reset', 'scope': scope, 'stats': stats})
    return {**result, 'stats': stats}


@app.get("/api/faculty")
async def get_faculty():
    return _dashboard_roster()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "fps":    state.fps_actual,
        "inference_fps": state.inference_fps,
        "hlc":    hlc_registry is not None,
        "ai":     config.ai.enabled,
        "attendance_mode": config.tracker.attendance_mode,
        "detector": getattr(config.model, "detector", "InsightFace"),
    }


@app.post("/api/audit/run")
async def run_audit(work_date: str = None):
    from datetime import date as date_cls
    target  = work_date or date_cls.today().isoformat()
    ev_loop = asyncio.get_event_loop()
    report  = await ev_loop.run_in_executor(
        None,
        lambda: audit_engine.run_daily_audit(target, "dashboard"),
    )
    return {
        "audit_date":    report.audit_date,
        "run_at":        report.run_at,
        "total_faculty": report.total_faculty,
        "clean":         report.clean,
        "corrected":     report.corrected,
        "flagged":       report.flagged,
        "absent_auto":   report.absent_auto,
        "results": [
            {
                "employee_id":      r.employee_id,
                "name":             r.name,
                "audit_status":     r.audit_status,
                "stored_final":     r.stored_final,
                "audit_final":      r.audit_final,
                "first_in":         r.first_in,
                "last_out":         r.last_out,
                "total_detections": r.total_detections,
                "corrections":      r.corrections,
                "flags":            r.flags,
            }
            for r in report.results
        ],
    }


@app.get("/api/audit/status")
async def audit_status(work_date: str = None):
    from datetime import date as date_cls
    target  = work_date or date_cls.today().isoformat()
    summary = audit_engine.get_audit_summary(target)
    return {"audited": summary is not None, "work_date": target, "summary": summary}


@app.get("/api/audit/flagged")
async def get_flagged(work_date: str = None):
    from datetime import date as date_cls
    target = work_date or date_cls.today().isoformat()
    return audit_engine.get_flagged_records(target)


@app.get("/api/export/attendance")
async def export_attendance(from_date: str = None, to_date: str = None,
                             employee_id: str = None):
    from datetime import date as date_cls
    today     = date_cls.today().isoformat()
    from_date = from_date or today
    to_date   = to_date   or today
    csv_data, warnings = db.export_csv(from_date, to_date, employee_id)
    if not csv_data:
        return Response(
            content="\n".join(warnings), media_type="text/plain",
            status_code=400, headers={"X-Audit-Required": "true"},
        )
    headers = {
        "Content-Disposition":
            f"attachment; filename=attendance_{from_date}_{to_date}.csv"
    }
    if warnings:
        headers["X-Audit-Warnings"] = " | ".join(warnings)
    return Response(content=csv_data, media_type="text/csv", headers=headers)


@app.get("/api/export/detections")
async def export_detections(from_date: str = None, to_date: str = None):
    from datetime import date as date_cls
    today     = date_cls.today().isoformat()
    from_date = from_date or today
    to_date   = to_date   or today
    csv_data  = db.export_detections_csv(from_date, to_date)
    return Response(
        content=csv_data, media_type="text/csv",
        headers={"Content-Disposition":
                  f"attachment; filename=detections_{from_date}_{to_date}.csv"},
    )


@app.get("/api/export/monthly")
async def export_monthly(year_month: str = None):
    from datetime import date as date_cls
    ym       = year_month or date_cls.today().strftime("%Y-%m")
    csv_data = db.export_monthly_csv(ym)
    return Response(
        content=csv_data, media_type="text/csv",
        headers={"Content-Disposition":
                  f"attachment; filename=monthly_{ym}.csv"},
    )


# ── HTML pages ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with (ROOT / "static" / "dashboard.html").open(encoding="utf-8") as f:
        html = f.read()
    if config.database.student_roster_ids is not None:
        html = html.replace("<body>", '<body class="student-mode">', 1)
    return html


@app.get("/hr-panel", response_class=HTMLResponse)
async def hr_panel():
    with (ROOT / "static" / "hr_admin.html").open(encoding="utf-8") as f:
        return f.read()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global main_attendance, hlc_proc, hlc_registry

    # ── Main attendance processor (staff room) ────────────────────────────────
    main_attendance = AttendanceProcessor(db, config)
    logger.info("Main attendance processor ready")

    # ── HLC processors (if enabled) ───────────────────────────────────────────
    if config.hlc.enabled:
        hlc_team_db    = HLCTeamDB(db_path=config.hlc.team_db_path)
        hlc_visitor_db = HLCVisitorDBV4(db_path=config.hlc.visitor_db_path)

        hlc_registry = build_registry_from_config(config)
        cam_ids      = hlc_registry.all_ids()
        logger.info(f"HLC cameras registered: {cam_ids}")

        # Start a CameraReader for each enabled HLC camera now, so they're
        # already connecting/buffering frames while the rest of startup
        # continues. The actual recognition loop for each starts further
        # below, once `engine` and the asyncio `loop` exist.
        global hlc_camera_readers
        hlc_camera_readers = {}
        for cam in hlc_registry.all_enabled():
            reader = CameraReader(rtsp_url=cam.rtsp_url, camera_id=cam.camera_id)
            reader.start()
            hlc_camera_readers[cam.camera_id] = reader
            logger.info(f"HLC camera reader started: {cam.camera_id} ({cam.label})")

        hlc_room_state = RoomStateManager()

        hlc_proc = HLCProcessorV4(
            camera_registry = hlc_registry,
            room_state      = hlc_room_state,
            team_db         = hlc_team_db,
            visitor_db      = hlc_visitor_db,
        )

        # Alert engine — 15-min overstay alerts
        hlc_alert_engine = HLCAlertEngine(
            visitor_db          = hlc_visitor_db,
            team_db             = hlc_team_db,
            check_interval_secs = 60,
            threshold_minutes   = config.hlc.alert_threshold_mins,
        )
        hlc_alert_engine.start()

        # Timeout sweep — auto-close stale sessions
        session_sweep = SessionTimeoutSweep(
            visitor_db          = hlc_visitor_db,
            timeout_hours       = config.hlc.session_timeout_hrs,
            check_interval_mins = 30,
        )
        session_sweep.start()

        # Inject into HLC API routes
        init_hlc(hlc_team_db, hlc_visitor_db, hlc_alert_engine)
        logger.info("HLC module ready")
    else:
        logger.info("HLC module disabled (config.hlc.enabled=False)")

    # ── AI module ─────────────────────────────────────────────────────────────
    if config.ai.enabled:
        init_ai(
            db_path        = config.database.db_path,
            hlc_team_db    = config.hlc.team_db_path,
            hlc_visitor_db = config.hlc.visitor_db_path,
            api_key        = config.ai.api_key,
        )
        key_status = "✓" if config.ai.api_key else "✗ set GEMINI_API_KEY in .env"
        logger.info(f"AI module ready | API key: {key_status}")

    # ── Camera + recognition ──────────────────────────────────────────────────
    if config.model.detector == "AdaFace original":
        from adaface_pipeline import AdaFaceRecognitionEngine
        engine = AdaFaceRecognitionEngine()
    elif config.model.detector == "YOLOv8 Face + InsightFace":
        from yolov8_face_recognition import YOLOv8FaceRecognitionEngine
        engine = YOLOv8FaceRecognitionEngine()
    else:
        engine = RecognitionEngine()
    if config.model.detector == "AdaFace original":
        from adaface_pipeline import AdaFaceCameraReader
        camera = AdaFaceCameraReader()
    else:
        camera = CameraReader()
    camera.start()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    from video_preview import VideoPreview
    preview = VideoPreview(
        camera,
        draw=lambda frame, tracks, people: _draw_overlays(
            frame, tracks, frame.shape[0], frame.shape[1], people),
        publish=lambda jpeg: setattr(state, 'latest_frame', jpeg),
        report_fps=lambda fps: setattr(state, 'fps_actual', fps),
        generation=lambda: state.reset_generation,
        max_fps=config.camera.fps,
    )
    def display_video():
        _server_ready.wait()
        preview.run()
    threading.Thread(target=display_video, daemon=True, name='VideoPreview').start()

    proc_thread = threading.Thread(
        target=processing_loop,
        args=(engine, camera, broadcast, loop, preview),
        daemon=True,
        name="ProcessingLoop",
    )
    proc_thread.start()

    # ── HLC camera processing loops (one thread per camera) ──────────────────
    for cam_id, reader in hlc_camera_readers.items():
        camera_role = "inside" if "INSIDE" in cam_id.upper() else "outside"
        t = threading.Thread(
            target=hlc_processing_loop,
            args=(cam_id, camera_role, reader, engine, broadcast, loop),
            daemon=True,
            name=f"HLCLoop-{cam_id}",
        )
        t.start()
        logger.info(f"HLC processing loop thread started: {cam_id} (role={camera_role})")

    try:
        uvicorn.run(
            app,
            host=config.dashboard.host,
            port=config.dashboard.port,
            log_level="warning",
            loop="asyncio",
        )
    finally:
        state.running = False
        preview.stop()
        camera.stop()


if __name__ == "__main__":
    main()
