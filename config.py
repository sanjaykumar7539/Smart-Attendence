"""
FaceTrack Pro — config.py
All runtime settings in one place. Edit this before running.

DEPLOYMENT TARGET: HP Pro Tower 280 G9 E (Intel i7-14700, Intel UHD 770
integrated graphics, Windows 11 Pro). No NVIDIA GPU — use_gpu forced False.
Camera decode runs on OpenCV+FFmpeg (GStreamer/NVDEC not used on this box).

DEMO PATCH CHANGELOG:
  detection_threshold   0.35 → 0.20   backlit/distant faces score 0.20-0.30
  recognition_threshold 0.28 → 0.22   more permissive for backlighting
  quality_threshold     0.35 → 0.20   luminance check was blocking dark faces
  min_face_size         30   → 20     catch smaller distant faces
  vote_confirm_n        3    → 2      faster confirmation
  vote_window_size      6    → 10     bigger window = more vote chances
  frame_skip            6    → 4      6fps AI instead of 4fps
  enable_liveness       True → False  WAS SILENTLY BLOCKING ALL RECOGNITION
  liveness_motion_min   1.2  → 0.3   safe value for future re-enable
  min_hits (tracker)    3    → 2      track activates faster (in tracker.py)
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class CameraConfig:
    # No separate third camera exists — reusing the real outside-facing
    # HLC camera for main attendance recognition too (per user: only 2
    # physical cameras total). Swap to .101 here if you'd rather the
    # INSIDE camera serve as the main "Live" tab feed instead.
    rtsp_url:     str  = "rtsp://admin:Admin%40123@127.0.0.1:8555/Streaming/Channels/101"
    # Input source: "rtsp" | "webcam" | "video"
    # For recorded classroom testing set input_mode="video" and video_path.
    input_mode: str = "video"

    use_webcam: bool = False
    webcam_index: int = 0

    # Classroom test video is stored in the project root.
    video_path: str = "./D10_20261009102035_1.mp4"
    video_loop: bool = False
    video_realtime: bool = True

    frame_width: int = 640
    frame_height: int = 480
    fps: int = 25
    frame_skip: int = 2

    camera_id: str = "CAM-0"
    location:     str  = "Staff Room Entrance"


@dataclass
class AttendancePolicy:
    on_time_arrival_cutoff:     str = "09:10"  # college day starts 9:10 — arrive at/before this = on-time
    permission_arrival_end:     str = "09:55"  # 9:10 + 45min permission grace
    afternoon_absent_cutoff:    str = "12:35"  # forenoon session ends here — missed forenoon entirely after this
    on_time_departure_start:    str = "16:10"  # college day ends 4:10 — leave at/after this = on-time
    permission_departure_start: str = "15:25"  # 16:10 - 45min permission grace
    half_day_cutoff:            str = "13:20"  # afternoon session starts here — leave before this = half day
    max_permissions_per_month:  int = 2


@dataclass
class ModelConfig:
    detector: str = "InsightFace"
    yolo_face_model_path: str = "./adaface_source/yolov8m-face.pt"
    yolo_face_image_size: int = 960
    yolo_face_confidence: float = 0.15
    yolo_face_iou: float = 0.45
    cpu_inference_threads: int = 2
    det_size: tuple = (320, 320)
    model_pack:            str   = "buffalo_s"
    recognition_threshold: float = 0.30   # was 0.22 (too permissive — caused false matches)
    detection_threshold:   float = 0.20   # PATCHED: was 0.35
    few_shot_matching:     bool  = True
    prototype_score_weight: float = 0.65
    detection_saturation_gain: float = 1.10  # gentle color lift before InsightFace detection
    detection_contrast_gain: float = 1.08    # slightly more contrast for washed-out frames
    detection_brightness_shift: int = 6       # tiny brightness bump for dim/flat footage
    min_margin:            float = 0.06   # top-1 vs top-2 similarity gap required to trust a match
    min_face_size:         int   = 20     # PATCHED: was 30
    use_gpu:               bool  = False  # HP Pro Tower has no NVIDIA GPU (was True on Jetson)
    gpu_id:                int   = 0
    models_dir:            str   = "./models"
    quality_threshold:     float = 0.20   # PATCHED: was 0.35
    vote_confirm_n:        int   = 2      # PATCHED: was 3
    vote_window_size:      int   = 10     # PATCHED: was 6
    enable_liveness:       bool  = False
    liveness_motion_min:   float = 0.3

    # ── Adaptive long-range recognition (CAM-01) ───────────────────────
    adaptive_recognition:   bool  = True
    face_buffer_size:       int   = 6
    recognition_vote_n:     int   = 2
    recognition_vote_window:int   = 6
    good_face_px:           int   = 96
    medium_face_px:         int   = 60
    blur_good:              float = 90.0
    blur_min:               float = 35.0
    brightness_min:         float = 45.0
    brightness_max:         float = 215.0
    threshold_good:         float = 0.30
    threshold_medium:       float = 0.33
    threshold_poor:         float = 0.37
    max_search_k:           int   = 8

    # Super-resolution is a fallback for small/poor faces only.
    enable_super_resolution: bool = True
    sr_model_path:          str   = "./models/realesrgan_x2.onnx"
    sr_scale:               int   = 2
    sr_trigger_face_px:     int   = 64
    sr_max_calls_per_track: int   = 1

    # Optional AdaFace verifier. It is enabled only when BOTH ONNX model and
    # an AdaFace gallery index exist. Otherwise ArcFace remains fully usable.
    enable_adaface:         bool  = True
    adaface_model_path:     str   = "./models/adaface_ir101.onnx"
    adaface_index_path:     str   = "./database/adaface_index.faiss"
    adaface_meta_path:      str   = "./database/adaface_meta.json"
    adaface_threshold:      float = 0.36
    adaface_weight:         float = 0.35


@dataclass
class TrackerConfig:
    attendance_mode: str = "line"  # student launcher uses presence
    max_age:              int   = 30
    min_hits:             int   = 2      # PATCHED: was 3
    iou_threshold:        float = 0.3
    line_position:        float = 0.57
    trigger_cooldown_sec: int   = 30


@dataclass
class DatabaseConfig:
    db_path:    str = "./database/attendance.db"
    index_path: str = "./database/faculty_index.faiss"
    meta_path:  str = "./database/faculty_meta.json"
    faces_dir:  str = "./faces"
    student_roster_ids: Optional[List[str]] = None


@dataclass
class DisplayConfig:
    show_bbox:        bool  = True
    show_landmarks:   bool  = False
    show_confidence:  bool  = True
    color_recognized: tuple = (0, 255, 120)
    color_unknown:    tuple = (0, 80, 255)


@dataclass
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 8001


# ── HLC (Horizon Learning Center) Config ─────────────────────────────────────
# Add your HLC camera URLs here once you get access to the cameras.
# Each camera needs: camera_id, door_id, rtsp_url, in_direction
#
# in_direction: which frame direction = person ENTERING the room
#   "top_to_bottom" → corridor at top of frame, room at bottom (most common)
#   "bottom_to_top" → room at top, corridor at bottom
#
# To calibrate: start the system, watch the stream, walk into HLC room,
# see which way your bounding box centroid moves → that is in_direction.

@dataclass
class HLCConfig:
    # Disabled while testing with a recorded MP4. Enable only when the two
    # physical HLC RTSP cameras are connected and their URLs are reachable.
    enabled:              bool  = False
    alert_threshold_mins: float = 15.0    # flag non-HLC-member if inside > 15 mins (was 5.0 — corrected 2026-08 to match actual policy)
    session_timeout_hrs:  float = 4.0     # auto-close stale sessions after 4h
    direction_frames:     int   = 3       # frames to confirm IN/OUT direction
    team_db_path:         str   = "hlc_team.db"
    visitor_db_path:      str   = "hlc_visitors.db"

    # ── Camera list — single door, two cameras (inside + outside), facing
    #    each other to catch both entry and exit reliably. Both share the
    #    SAME door_id. TODO: replace both rtsp_url values with your real
    #    camera URLs once you have them — everything else here is already
    #    correctly set up for this topology.
    #
    #    in_direction: describes which way movement reads as "entering" for
    #    THAT camera's video orientation. Since the two cameras face each
    #    other across the same doorway, a person walking IN moves in
    #    OPPOSITE visual directions for each — that's why these two values
    #    are set as opposites below. Verify/adjust after your first live
    #    test: watch someone walk in, and if the system logs it as OUT,
    #    just flip that camera's in_direction.
    cameras: List[Dict] = field(default_factory=lambda: [
        {
            "camera_id":      "HLC-CAM-OUTSIDE",
            "door_id":        "DOOR_MAIN",
            # Real camera — sub-stream (102) used to reduce CPU load
            # (3 concurrent AI pipelines now run on this box)
            "rtsp_url":       "rtsp://admin:Admin%40123@127.0.0.1:8555/Streaming/Channels/101",
            "in_direction":   "top_to_bottom",   # adjust after calibration — see note above
            "label":          "Main Door - Outside",
            "confirm_frames": 3,
            "enabled":        False,
        },
        {
            "camera_id":      "HLC-CAM-INSIDE",
            "door_id":        "DOOR_MAIN",
            "rtsp_url":       "rtsp://admin:Admin%40123@127.0.0.1:8554/Streaming/Channels/101",
            "in_direction":   "bottom_to_top",   # opposite of the outside camera — faces the other way
            "label":          "Main Door - Inside",
            "confirm_frames": 3,
            "enabled":        False,
        },
    ])


# ── HLC Alert Notification Channels ──────────────────────────────────────────
# Phase 1 (now): browser/dashboard push via the Notification API — works
# whenever the dashboard tab is open (no server credentials needed).
# Phase 2 (later): flip these on once you have the credentials. The alert
# engine already writes to alert_log with the right channel name either way,
# so turning these on later needs no schema change — just fill in the
# creds and set enabled=True.

@dataclass
class WebPushConfig:
    enabled: bool = True   # dashboard Notification API — on by default, no setup needed


@dataclass
class EmailAlertConfig:
    enabled:    bool = False   # flip on once SMTP creds are set below
    smtp_host:  str  = ""
    smtp_port:  int  = 587
    smtp_user:  str  = ""
    smtp_pass:  str  = field(default_factory=lambda: os.getenv("HLC_ALERT_SMTP_PASS", ""))
    from_addr:  str  = ""
    to_addrs:   List[str] = field(default_factory=list)   # e.g. ["principal@egspec.org"]


@dataclass
class WhatsAppAlertConfig:
    enabled:      bool = False   # flip on once WhatsApp API creds are set below
    provider:     str  = "twilio"   # "twilio" | "business_api"
    account_sid:  str  = field(default_factory=lambda: os.getenv("HLC_ALERT_WA_SID", ""))
    auth_token:   str  = field(default_factory=lambda: os.getenv("HLC_ALERT_WA_TOKEN", ""))
    from_number:  str  = ""              # e.g. "whatsapp:+14155238886"
    to_numbers:   List[str] = field(default_factory=list)   # e.g. ["whatsapp:+9199xxxxxxx"]


@dataclass
class NotifyConfig:
    web_push: WebPushConfig       = field(default_factory=WebPushConfig)
    email:    EmailAlertConfig    = field(default_factory=EmailAlertConfig)
    whatsapp: WhatsAppAlertConfig = field(default_factory=WhatsAppAlertConfig)


# ── AI Module Config ──────────────────────────────────────────────────────────
# Get your free API key at: https://aistudio.google.com/apikey
# Set it in .env file: GEMINI_API_KEY=AIza...
# OR set directly below (not recommended for production)

@dataclass
class AIConfig:
    enabled:        bool = True
    api_key:        str  = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model:          str  = "gemini-2.5-flash"


# ── Root Config ───────────────────────────────────────────────────────────────

@dataclass
class Config:
    camera:    CameraConfig     = field(default_factory=CameraConfig)
    policy:    AttendancePolicy = field(default_factory=AttendancePolicy)
    model:     ModelConfig      = field(default_factory=ModelConfig)
    tracker:   TrackerConfig    = field(default_factory=TrackerConfig)
    database:  DatabaseConfig   = field(default_factory=DatabaseConfig)
    display:   DisplayConfig    = field(default_factory=DisplayConfig)
    dashboard: DashboardConfig  = field(default_factory=DashboardConfig)
    hlc:       HLCConfig        = field(default_factory=HLCConfig)
    notify:    NotifyConfig     = field(default_factory=NotifyConfig)
    ai:        AIConfig         = field(default_factory=AIConfig)

    college_name: str = "EGS Pillay Group of Institutions, Nagapattinam"


config = Config()
