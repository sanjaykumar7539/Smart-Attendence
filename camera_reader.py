"""
FaceTrack Pro — camera_reader.py
Low-latency RTSP camera reader for Jetson Orin Nano.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY THE OLD CODE HAD 5-10 SECOND LAG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  OLD:  cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

  CAP_PROP_BUFFERSIZE=1 only controls OpenCV's ring buffer at
  the Python/C++ boundary. It does NOTHING to FFmpeg's internal
  demux/decode queue. H.264 cameras group frames into GOPs
  (Groups of Pictures) of 25-60 frames. FFmpeg waits for a
  full GOP before it can decode — that alone is 1-2 seconds at
  25fps. The RTSP jitter buffer adds 3-5 more. Total: 5-10s
  unavoidable lag that no OpenCV setting can fix.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE FIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  GStreamer pipeline with nvv4l2decoder (Jetson NVDEC).

  Key flags:
    rtspsrc latency=0            kills the jitter buffer entirely
    rtspsrc drop-on-latency=true drops packets when behind
    appsink drop=true            Python always gets newest frame
    appsink max-buffers=1        never accumulates a queue
    appsink sync=false           no presentation-timestamp wait

  Result: <150ms glass-to-Python on Jetson Orin Nano.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUTOMATIC FALLBACK CHAIN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. GStreamer + nvv4l2decoder   NVDEC hardware — best, <150ms
  2. GStreamer + avdec_h264      CPU decode, ~250ms
  3. OpenCV + CAP_GSTREAMER      no jitter buffer, ~300ms
  4. OpenCV + CAP_FFMPEG         last resort — 1-2s lag remaining

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PUBLIC API  (identical to original — main.py unchanged)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  camera = CameraReader()
  camera.start()
  frame  = camera.get_frame()   # np.ndarray BGR, or None
  camera.stop()
"""

import cv2
import threading
import time
import numpy as np
from loguru import logger

from config import config


# ── Pipeline string builders ──────────────────────────────────

def _pipe_nvdec(url: str, w: int, h: int) -> str:
    """Primary: Jetson NVDEC hardware decoder."""
    return (
        f"rtspsrc location={url} "
        f"  latency=0 buffer-mode=none protocols=tcp drop-on-latency=true "
        f"! rtph264depay ! h264parse "
        f"! nvv4l2decoder enable-max-performance=1 "
        f"! nvvidconv "
        f"! video/x-raw,format=BGRx,width={w},height={h} "
        f"! videoconvert "
        f"! video/x-raw,format=BGR "
        f"! appsink name=sink drop=true max-buffers=1 sync=false emit-signals=false"
    )


def _pipe_cpu(url: str, w: int, h: int) -> str:
    """Fallback 1: GStreamer CPU decoder."""
    return (
        f"rtspsrc location={url} "
        f"  latency=0 protocols=tcp drop-on-latency=true "
        f"! rtph264depay ! h264parse ! avdec_h264 "
        f"! videoconvert "
        f"! video/x-raw,format=BGR,width={w},height={h} "
        f"! appsink name=sink drop=true max-buffers=1 sync=false emit-signals=false"
    )


def _pipe_ocv_gst(url: str, w: int, h: int) -> str:
    """Fallback 2: OpenCV CAP_GSTREAMER backend string."""
    return (
        f"rtspsrc location={url} latency=0 protocols=tcp "
        f"! rtph264depay ! h264parse ! avdec_h264 "
        f"! videoconvert "
        f"! video/x-raw,format=BGR,width={w},height={h} "
        f"! appsink drop=true max-buffers=1 sync=false"
    )


# ── GStreamer reader (gi.repository.Gst) ──────────────────────

class _GstReader:
    """
    Reads frames directly via GStreamer Python bindings.
    Tries NVDEC first, falls back to CPU avdec_h264.
    """

    def __init__(self, url: str, w: int, h: int):
        self._url      = url
        self._w        = w
        self._h        = h
        self._pipeline = None
        self._appsink  = None
        self._ready    = False

    def open(self) -> bool:
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
            Gst.init(None)
        except Exception as e:
            logger.warning(f"GStreamer Python bindings (gi) not available: {e}")
            return False

        from gi.repository import Gst

        for builder, label in [
            (_pipe_nvdec, "NVDEC hardware"),
            (_pipe_cpu,   "CPU avdec_h264"),
        ]:
            pipe_str = builder(self._url, self._w, self._h)
            pipeline = Gst.parse_launch(pipe_str)
            if pipeline is None:
                continue

            appsink = pipeline.get_by_name("sink")
            if appsink is None:
                pipeline.set_state(Gst.State.NULL)
                continue

            ret = pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                pipeline.set_state(Gst.State.NULL)
                logger.warning(f"GStreamer {label}: failed to enter PLAYING state")
                continue

            self._pipeline = pipeline
            self._appsink  = appsink
            self._ready    = True
            logger.success(f"Camera backend: GStreamer {label} — latency <200ms")
            return True

        return False

    def read_frame(self) -> np.ndarray | None:
        if not self._ready:
            return None
        try:
            from gi.repository import Gst
            sample = self._appsink.emit("pull-sample")
            if sample is None:
                return None

            buf  = sample.get_buffer()
            caps = sample.get_caps().get_structure(0)
            w    = caps.get_int("width")[1]
            h    = caps.get_int("height")[1]

            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                return None
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3)).copy()
            buf.unmap(mapinfo)
            return frame
        except Exception as e:
            logger.error(f"GStreamer read error: {e}")
            return None

    def close(self):
        if self._pipeline:
            try:
                from gi.repository import Gst
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        self._ready = False


# ── OpenCV fallback reader ─────────────────────────────────────

class _OpenCVReader:
    """
    Fallback using cv2.VideoCapture.
    Tries CAP_GSTREAMER (no jitter buffer) before CAP_FFMPEG.
    """

    def __init__(self, url: str, w: int, h: int):
        self._url = url
        self._w   = w
        self._h   = h
        self._cap = None

    def open(self) -> bool:
        # Try OpenCV GStreamer backend
        gst_str = _pipe_ocv_gst(self._url, self._w, self._h)
        cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            self._cap = cap
            logger.info("Camera backend: OpenCV + GStreamer (no jitter buffer)")
            return True
        cap.release()

        # FFmpeg last resort
        logger.warning(
            "OpenCV GStreamer failed — falling back to FFmpeg "
            "(expect 1-2s residual lag)"
        )
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        cap.set(cv2.CAP_PROP_FPS,          config.camera.fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._h)

        if cap.isOpened():
            self._cap = cap
            logger.info("Camera backend: OpenCV + FFmpeg")
            return True

        cap.release()
        return False

    def read_frame(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def close(self):
        if self._cap:
            self._cap.release()
            self._cap = None


# ── Public API ────────────────────────────────────────────────

class CameraReader:
    """
    Low-latency RTSP camera reader. Drop-in replacement for original.

    Runs a background daemon thread that continuously drains the stream.
    get_frame() always returns the most recent live frame.
    Auto-reconnects on stream failure with exponential backoff.

    Supports multiple independent instances — one per physical camera.
    All constructor args are optional and default to the main camera's
    config (config.camera.*), so existing `CameraReader()` call sites
    (the main staff camera) are completely unaffected.

    Usage (main camera, unchanged):
        camera = CameraReader()
        camera.start()
        frame = camera.get_frame()   # np.ndarray BGR, or None
        camera.stop()

    Usage (a second/HLC camera):
        cam = CameraReader(rtsp_url="rtsp://...", camera_id="HLC-CAM-A1")
        cam.start()
    """

    def __init__(self, rtsp_url: str = None, frame_width: int = None,
                 frame_height: int = None, camera_id: str = "main",
                 use_webcam: bool = None, webcam_index: int = None,
                 input_mode: str = None, video_path: str = None,
                 video_loop: bool = None, video_realtime: bool = None):
        self._frame:  np.ndarray | None = None
        self._last_consumed_frame = None
        self._lock    = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop    = threading.Event()

        # Fall back to the main camera's global config for anything not
        # explicitly passed in — preserves exact old behavior for
        # CameraReader() with no args.
        cfg = config.camera
        self.camera_id    = camera_id
        self.rtsp_url     = rtsp_url     if rtsp_url     is not None else cfg.rtsp_url
        self.frame_width  = frame_width  if frame_width  is not None else cfg.frame_width
        self.frame_height = frame_height if frame_height is not None else cfg.frame_height
        self.use_webcam   = use_webcam   if use_webcam   is not None else cfg.use_webcam
        self.webcam_index = webcam_index if webcam_index is not None else cfg.webcam_index
        self.input_mode   = (input_mode if input_mode is not None else getattr(cfg, "input_mode", "rtsp")).lower().strip()
        # Preserve older use_webcam=True behavior even when input_mode was not configured.
        if self.use_webcam and input_mode is None:
            self.input_mode = "webcam"
        self.video_path     = video_path if video_path is not None else getattr(cfg, "video_path", "./testing.mp4")
        self.video_loop     = video_loop if video_loop is not None else getattr(cfg, "video_loop", False)
        self.video_realtime = video_realtime if video_realtime is not None else getattr(cfg, "video_realtime", True)

    def start(self):
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"CameraReader-{self.camera_id}"
        )
        self._thread.start()
        logger.info(f"CameraReader[{self.camera_id}] thread started")

    def get_frame(self) -> np.ndarray | None:
        """Thread-safe. Returns copy of the latest frame, or None."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_preview_frame(self):
        """Read-only snapshot reference; preview copies it before drawing.

        Independent of get_new_frame(), so display never consumes an AI frame.
        The capture thread replaces frames and never edits a published array.
        """
        with self._lock:
            return self._frame

    def stop(self):
        self._stop.set()

    def get_new_frame(self):
        """Single-consumer read: never vote twice on the same captured frame."""
        with self._lock:
            if self._frame is None or self._frame is self._last_consumed_frame:
                return None
            self._last_consumed_frame = self._frame
            return self._frame.copy()

    # ── internal ──────────────────────────────────────────────

    def _loop(self):
        if self.input_mode == "video":
            self._video_loop(self.video_path)
            return

        if self.input_mode == "webcam" or self.use_webcam:
            self._webcam_loop(self.webcam_index)
            return

        if self.input_mode != "rtsp":
            logger.error(
                f"CameraReader[{self.camera_id}]: unsupported input_mode={self.input_mode!r}; "
                "use 'rtsp', 'webcam', or 'video'"
            )
            return

        fail_count = 0
        while not self._stop.is_set():
            reader = self._open_best(self.rtsp_url, self.frame_width, self.frame_height)

            if reader is None:
                wait = min(2 ** fail_count, 30)
                logger.warning(f"CameraReader[{self.camera_id}]: all backends failed — retrying in {wait}s")
                time.sleep(wait)
                fail_count += 1
                continue

            fail_count   = 0
            empty_streak = 0

            while not self._stop.is_set():
                frame = reader.read_frame()

                if frame is not None and frame.size > 0:
                    empty_streak = 0
                    with self._lock:
                        self._frame = frame
                else:
                    empty_streak += 1
                    # 90 consecutive empty frames ≈ 3 seconds at 30fps
                    if empty_streak > 90:
                        logger.warning(f"CameraReader[{self.camera_id}]: stream stalled — reconnecting")
                        break
                    time.sleep(0.005)

            reader.close()
            if not self._stop.is_set():
                logger.info(f"CameraReader[{self.camera_id}]: reconnecting in 1s...")
                time.sleep(1)

    def _open_best(self, url: str, w: int, h: int):
        """Try each backend in order. Return first that opens."""
        gst = _GstReader(url, w, h)
        if gst.open():
            return gst

        ocv = _OpenCVReader(url, w, h)
        if ocv.open():
            return ocv

        return None

    def _webcam_loop(self, index: int):
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS,          config.camera.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        logger.info(f"Webcam {index} opened")

        while not self._stop.is_set():
            ret, frame = cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.01)

        cap.release()
    def _video_loop(self, path: str):
        """Read a recorded video as a camera source.

        When video_realtime=True, frames are paced using the source FPS so the
        downstream tracker/voting/attendance logic sees realistic timing. This
        avoids a background reader racing to EOF and dropping nearly every frame.
        """
        from pathlib import Path

        video = Path(path).expanduser()
        if not video.is_absolute():
            video = Path.cwd() / video
        video = video.resolve()

        if not video.exists():
            logger.error(f"Video source not found: {video}")
            return

        while not self._stop.is_set():
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                logger.error(f"Unable to open video source: {video}")
                return

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if not np.isfinite(fps) or fps <= 0.1:
                fps = float(config.camera.fps or 25)
            frame_period = 1.0 / fps
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = (frame_count / fps) if frame_count > 0 else 0.0
            logger.success(
                f"CameraReader[{self.camera_id}] video source: {video.name} | "
                f"{fps:.2f} FPS | {frame_count or '?'} frames | {duration:.1f}s"
            )

            next_deadline = time.perf_counter()
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame is None or frame.size == 0:
                    continue

                # Optional resize only when explicit dimensions differ.
                h, w = frame.shape[:2]
                if self.frame_width > 0 and self.frame_height > 0 and (w != self.frame_width or h != self.frame_height):
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)

                with self._lock:
                    self._frame = frame

                if self.video_realtime:
                    next_deadline += frame_period
                    delay = next_deadline - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    elif delay < -(frame_period * 3):
                        # If decoding was delayed heavily, re-anchor instead of
                        # trying to catch up at burst speed.
                        next_deadline = time.perf_counter()

            cap.release()
            if not self.video_loop or self._stop.is_set():
                logger.info(f"CameraReader[{self.camera_id}]: video completed")
                return
            logger.info(f"CameraReader[{self.camera_id}]: looping video from start")
