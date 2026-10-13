"""Display the newest camera frame without waiting for recognition inference."""
from dataclasses import dataclass, replace
import threading
import time

import cv2
import numpy as np


def motion_gray(frame):
    height, width = frame.shape[:2]
    scale = min(1.0, 640 / width)
    small = cv2.resize(frame, (round(width * scale), round(height * scale)))
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def move_overlays(previous, current, tracks, people, frame_shape):
    """Follow image features between AI results; never create or confirm identities."""
    if previous is None or previous.shape != current.shape:
        return (), ()
    height, width = frame_shape[:2]
    sx, sy = current.shape[1] / width, current.shape[0] / height
    boxes = [track.bbox for track in tracks] + list(people)
    points, ranges = [], []
    offset = 0
    for x1, y1, x2, y2 in boxes:
        left, top = max(0, int(x1*sx)), max(0, int(y1*sy))
        right, bottom = min(current.shape[1], int(x2*sx)), min(current.shape[0], int(y2*sy))
        crop = previous[top:bottom, left:right]
        features = None
        if crop.shape[0] >= 5 and crop.shape[1] >= 5:
            features = cv2.goodFeaturesToTrack(crop, maxCorners=16, qualityLevel=0.01,
                                               minDistance=3, blockSize=3)
        count = 0 if features is None else len(features)
        if count:
            points.append(features + np.array([[[left, top]]], np.float32))
        ranges.append((offset, offset+count))
        offset += count
    if not points:
        return tuple(tracks), tuple(people)
    old = np.concatenate(points).astype(np.float32)
    options = dict(winSize=(21,21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    new, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, old, None, **options)
    if new is None:
        return tuple(tracks), tuple(people)
    back, reverse_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, new, None, **options)
    if back is None:
        return tuple(tracks), tuple(people)
    valid = (status.ravel() == 1) & (reverse_status.ravel() == 1)
    valid &= np.linalg.norm((back-old).reshape(-1,2), axis=1) < 1.5
    displacements = (new-old).reshape(-1,2)
    output_tracks, output_people = [], []
    for index, ((start, end), box) in enumerate(zip(ranges, boxes)):
        vectors = displacements[start:end][valid[start:end]]
        if len(vectors) < 3:
            shifted = np.array(box)
        else:
            delta = np.median(vectors, axis=0)
            inliers = vectors[np.linalg.norm(vectors-delta, axis=1) < 3.0]
            if len(inliers) < max(3, len(vectors)//2):
                shifted = np.array(box)
            else:
                dx, dy = np.median(inliers, axis=0) / [sx, sy]
                shifted = np.array(box) + [dx, dy, dx, dy]
        shifted[[0,2]] = np.clip(shifted[[0,2]], 0, width-1)
        shifted[[1,3]] = np.clip(shifted[[1,3]], 0, height-1)
        if shifted[2] <= shifted[0] or shifted[3] <= shifted[1]:
            continue
        if index < len(tracks):
            output_tracks.append(replace(tracks[index], bbox=tuple(shifted)))
        else:
            output_people.append(tuple(shifted))
    return tuple(output_tracks), tuple(output_people)


@dataclass(frozen=True)
class OverlayTrack:
    bbox: tuple
    name: str
    employee_id: str | None
    recognized: bool
    recognition_status: str
    confidence: float
    no_losses: int

    def get_bbox(self):
        return np.asarray(self.bbox)


class VideoPreview:
    def __init__(self, camera, draw, publish, report_fps, generation, max_fps=20):
        self.camera = camera
        self.draw = draw
        self.publish = publish
        self.report_fps = report_fps
        self.generation = generation
        self.period = 1.0 / max(1, max_fps)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._overlay = ((), (), 0.0, -1, None)

    def update_overlays(self, tracks, people, generation, source_frame=None, source_time=None):
        # Copy scalar state; the inference thread keeps mutating its Kalman tracks.
        frozen = tuple(OverlayTrack(tuple(t.get_bbox()), t.name, t.employee_id,
                                   t.recognized, t.recognition_status,
                                   t.confidence, t.no_losses) for t in tracks)
        gray = motion_gray(source_frame) if source_frame is not None else None
        with self._lock:
            self._overlay = (frozen, tuple(tuple(box) for box in people),
                             time.monotonic(), generation, gray)

    def stop(self):
        self._stop.set()

    def run(self):
        last_source = None
        next_frame = 0.0
        fps_start, frames = time.monotonic(), 0
        overlay_version = None
        tracked, tracked_people, previous_gray = (), (), None
        while not self._stop.is_set():
            now = time.monotonic()
            if now - fps_start >= 1.0:
                self.report_fps(frames / (now - fps_start))
                fps_start, frames = now, 0
            source = self.camera.get_preview_frame()
            if source is None or source is last_source or now < next_frame:
                self._stop.wait(0.003)
                continue
            last_source = source
            next_frame = now + self.period
            frame = source.copy()
            gray = motion_gray(source)
            with self._lock:
                snapshot = self._overlay
            tracks, people, updated_at, generation, source_gray = snapshot
            if snapshot is not overlay_version:
                tracked, tracked_people, previous_gray = tracks, people, source_gray
                overlay_version = snapshot
            if previous_gray is not None and (tracked or tracked_people):
                tracked, tracked_people = move_overlays(previous_gray, gray, tracked, tracked_people, frame.shape)
            previous_gray = gray
            # Do not leave labels frozen indefinitely if inference stops or resets.
            if now - updated_at > 15.0 or generation != self.generation():
                tracked, tracked_people = (), ()
            self.draw(frame, tracked, tracked_people)
            ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                self.publish(jpeg.tobytes())
                frames += 1
