"""
hlc_camera_config.py
─────────────────────
Camera registry for HLC multi-door setup.

Each camera is assigned to a specific door and knows:
  - door_id        : which physical door it watches ("DOOR_A" / "DOOR_B")
  - line_position  : where the virtual crossing line sits in the frame (0.0–1.0)
  - in_direction   : which direction of crossing means "entering the room"
                     "top_to_bottom" → centroid.y increases = person walking into room
                     "bottom_to_top" → centroid.y decreases = person walking into room

Camera placement → in_direction mapping:
  ┌────────────────────────────────────────────────────────────────┐
  │  Camera mounted above door frame, angled DOWN toward threshold  │
  │  Corridor is at TOP of frame, Room interior is at BOTTOM        │
  │  Person entering: moves top→bottom → in_direction="top_to_bottom"│
  │                                                                 │
  │  Camera mounted inside room, angled toward door                 │
  │  Room interior is at TOP of frame, Corridor at BOTTOM           │
  │  Person entering: moves bottom→top → in_direction="bottom_to_top"│
  └────────────────────────────────────────────────────────────────┘

You can also override per-camera using left/right (for side-mounted):
  in_direction = "left_to_right" or "right_to_left"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class HLCCamera:
    camera_id:      str           # e.g. "HLC-CAM-A1"
    door_id:        str           # e.g. "DOOR_A"
    rtsp_url:       str           # camera stream URL
    in_direction:   str           # "top_to_bottom" | "bottom_to_top" | "left_to_right" | "right_to_left"
    line_position:  float = 0.5   # crossing line at 50% of frame by default
    confirm_frames: int   = 3     # frames needed to confirm crossing
    label:          str   = ""    # human label, e.g. "Door A – outer cam"
    enabled:        bool  = True


class HLCCameraRegistry:
    """
    Holds all HLC camera configurations.
    Loaded from config.py / environment variables at startup.

    Usage:
        registry = HLCCameraRegistry()
        registry.register(HLCCamera(
            camera_id    = "HLC-CAM-A1",
            door_id      = "DOOR_A",
            rtsp_url     = "rtsp://admin:pass@192.168.0.101:554/stream",
            in_direction = "top_to_bottom",
            label        = "Door A – corridor side",
        ))
        cam = registry.get("HLC-CAM-A1")
        # cam.is_entering(direction="top_to_bottom") → True
    """

    def __init__(self):
        self._cameras: Dict[str, HLCCamera] = {}

    def register(self, cam: HLCCamera):
        self._cameras[cam.camera_id] = cam

    def get(self, camera_id: str) -> Optional[HLCCamera]:
        return self._cameras.get(camera_id)

    def all_enabled(self) -> List[HLCCamera]:
        return [c for c in self._cameras.values() if c.enabled]

    def all_ids(self) -> List[str]:
        return [c.camera_id for c in self.all_enabled()]

    def is_hlc_camera(self, camera_id: str) -> bool:
        cam = self.get(camera_id)
        return cam is not None and cam.enabled

    def resolve_room_direction(self, camera_id: str,
                                detected_direction: str) -> Optional[str]:
        """
        Convert a camera-frame direction into a room direction.

        detected_direction: what the tracker reports ("top_to_bottom",
                            "bottom_to_top", "left_to_right", "right_to_left",
                            or legacy "IN"/"OUT" if tracker already resolved it)
        Returns: "IN" (entering room) | "OUT" (leaving room) | None (unknown)

        This is the KEY function that makes multi-door work correctly.
        Each camera knows what its "in" direction is, so the same physical
        movement (e.g. top→bottom) means different things for different cameras
        depending on where they are mounted.
        """
        cam = self.get(camera_id)
        if not cam:
            return None

        # Legacy passthrough (tracker already resolved to IN/OUT)
        if detected_direction in ("IN", "OUT"):
            return detected_direction

        if detected_direction == cam.in_direction:
            return "IN"
        else:
            return "OUT"


# ─────────────────────────────────────────────────────────────────────────────
# Default registry builder — called from main.py
# Reads camera configs from config.py / environment
# ─────────────────────────────────────────────────────────────────────────────

def build_registry_from_config(config) -> HLCCameraRegistry:
    """
    Build the camera registry from your config object.

    In config.py, define:

        hlc_cameras = [
            {
                "camera_id":    "HLC-CAM-A1",
                "door_id":      "DOOR_A",
                "rtsp_url":     "rtsp://admin:pass@192.168.0.101:554/stream",
                "in_direction": "top_to_bottom",
                "label":        "Door A – outer",
            },
            {
                "camera_id":    "HLC-CAM-A2",
                "door_id":      "DOOR_A",
                "rtsp_url":     "rtsp://admin:pass@192.168.0.102:554/stream",
                "in_direction": "bottom_to_top",
                "label":        "Door A – inner (optional redundancy)",
            },
            {
                "camera_id":    "HLC-CAM-B1",
                "door_id":      "DOOR_B",
                "rtsp_url":     "rtsp://admin:pass@192.168.0.103:554/stream",
                "in_direction": "top_to_bottom",
                "label":        "Door B – outer",
            },
            {
                "camera_id":    "HLC-CAM-B2",
                "door_id":      "DOOR_B",
                "rtsp_url":     "rtsp://admin:pass@192.168.0.104:554/stream",
                "in_direction": "bottom_to_top",
                "label":        "Door B – inner (optional redundancy)",
            },
        ]
    """
    registry = HLCCameraRegistry()
    cam_list = getattr(config.hlc, "cameras", [])

    for cfg in cam_list:
        cam = HLCCamera(
            camera_id     = cfg["camera_id"],
            door_id       = cfg["door_id"],
            rtsp_url      = cfg["rtsp_url"],
            in_direction  = cfg["in_direction"],
            line_position = cfg.get("line_position", 0.5),
            confirm_frames= cfg.get("confirm_frames", 3),
            label         = cfg.get("label", cfg["camera_id"]),
            enabled       = cfg.get("enabled", True),
        )
        registry.register(cam)

    return registry


# ─────────────────────────────────────────────────────────────────────────────
# Minimal config.py additions to paste in
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_ADDITIONS = """
# ── Add to config.py ─────────────────────────────────────────────────────────

hlc_cameras = [
    {
        "camera_id":    "HLC-CAM-A1",
        "door_id":      "DOOR_A",
        "rtsp_url":     "rtsp://admin:password@192.168.0.101:554/Streaming/Channels/101",
        "in_direction": "top_to_bottom",   # corridor at top, room at bottom
        "label":        "Door A",
        "confirm_frames": 3,
    },
    {
        "camera_id":    "HLC-CAM-B1",
        "door_id":      "DOOR_B",
        "rtsp_url":     "rtsp://admin:password@192.168.0.102:554/Streaming/Channels/101",
        "in_direction": "top_to_bottom",   # adjust if mounted differently
        "label":        "Door B",
        "confirm_frames": 3,
    },
    # Add 3rd and 4th cameras here if needed:
    # {
    #     "camera_id":    "HLC-CAM-A2",
    #     "door_id":      "DOOR_A",
    #     "rtsp_url":     "rtsp://admin:password@192.168.0.103:554/...",
    #     "in_direction": "bottom_to_top",  # inner camera faces opposite way
    #     "label":        "Door A – inner cam",
    # },
]

hlc_session_timeout_hrs  = 4.0
hlc_direction_frames     = 3
hlc_alert_threshold_mins = 15
"""
