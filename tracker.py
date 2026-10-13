"""
FaceTrack Pro — tracker.py
SORT: Simple Online and Realtime Tracking
Kalman Filter + Hungarian Algorithm + IOU Matching
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou(bb_test: np.ndarray, bb_gt: np.ndarray) -> float:
    """Compute IOU between two bounding boxes [x1,y1,x2,y2]."""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])

    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h

    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt   = (bb_gt[2]   - bb_gt[0])   * (bb_gt[3]   - bb_gt[1])
    union     = area_test + area_gt - inter

    return inter / union if union > 0 else 0.0


class KalmanFaceTrack:
    """Single face track with Kalman filter state."""

    count = 0   # Class-level track ID counter

    def __init__(self, bbox: np.ndarray):
        # State vector: [cx, cy, w, h, vx, vy, vw, vh]
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)

        self.kf.H         = np.eye(4, 8, dtype=float)
        self.kf.R[2:, 2:] *= 10
        self.kf.P[4:, 4:] *= 1000
        self.kf.P         *= 10
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4]      = self._to_state(bbox)

        KalmanFaceTrack.count += 1
        self.track_id  = KalmanFaceTrack.count
        self.hits      = 1
        self.no_losses = 0
        self.age       = 1

        # ── Recognition state ──────────────────────────────────
        # FIX-L5: recognized is STICKY — once set True, never reverts.
        # A single bad-angle frame cannot flip a confirmed ID back to Unknown.
        self.employee_id: str   = None
        self.name:        str   = "Unknown"
        self.confidence:  float = 0.0
        self.recognized:  bool  = False
        self.recognition_status: str = "UNKNOWN"
        self.quality_level: str = "unknown"
        self.was_enhanced: bool = False
        self.embedding = None

        # ── Line crossing state ────────────────────────────────
        # FIX-L4: prev_cy is ONLY updated AFTER direction is computed.
        # Old code overwrote prev_cy first → direction always "OUT".
        self.prev_cy:        float = None
        self.last_direction: str   = "IN"

    def _to_state(self, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return np.array(
            [[(x1 + x2) / 2], [(y1 + y2) / 2], [x2 - x1], [y2 - y1]],
            dtype=float,
        )

    def predict(self) -> np.ndarray:
        self.kf.predict()
        self.age += 1
        if self.no_losses > 0:
            self.hits = 0
        self.no_losses += 1
        return self._to_bbox()

    def update(self, bbox: np.ndarray):
        self.no_losses = 0
        self.hits += 1
        self.kf.update(self._to_state(bbox))

    def _to_bbox(self) -> np.ndarray:
        cx, cy, w, h = self.kf.x[:4].flatten()
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def get_center_y(self) -> float:
        return float(self.kf.x[1, 0])

    def get_bbox(self) -> np.ndarray:
        return self._to_bbox()


class SORTTracker:
    """
    SORT: Simple Online Realtime Tracker.
    Manages multiple KalmanFaceTrack objects across frames.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.3):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: list  = []
        self.frame_count   = 0

    def update(self, detections: list) -> list:
        """
        Update tracker with new detections.

        Args:
            detections: list of dicts with keys:
                        'bbox' [x1,y1,x2,y2], 'confidence'
        Returns:
            list of active confirmed KalmanFaceTrack objects
        """
        self.frame_count += 1

        # Predict new positions for all existing tracks
        for track in self.tracks:
            track.predict()

        if not detections:
            self._remove_dead()
            return self._active()

        det_boxes = np.array([d["bbox"] for d in detections])

        if self.tracks:
            trk_boxes = np.array([t.get_bbox() for t in self.tracks])
            matched, unmatched_dets, unmatched_tracks = self._match(det_boxes, trk_boxes)
            # Recover short motion gaps using face appearance, never attendance state.
            recovered = self._recover(detections, unmatched_dets, unmatched_tracks)
            matched.extend(recovered)
            recovered_dets = {d for _, d in recovered}
            unmatched_dets = [d for d in unmatched_dets if d not in recovered_dets]
            for t_idx, d_idx in matched:
                self.tracks[t_idx].update(det_boxes[d_idx])
                self._remember(self.tracks[t_idx], detections[d_idx])
        else:
            unmatched_dets = list(range(len(detections)))

        # Spawn new tracks for unmatched detections
        for d_idx in unmatched_dets:
            track = KalmanFaceTrack(det_boxes[d_idx])
            self._remember(track, detections[d_idx])
            self.tracks.append(track)

        self._remove_dead()
        return self._active()

    @staticmethod
    def _embedding(detection):
        value = detection.get('embedding')
        if value is None:
            return None
        value = np.asarray(value, dtype=float).reshape(-1)
        norm = np.linalg.norm(value)
        return value / norm if np.isfinite(norm) and norm > 1e-8 else None

    def _remember(self, track, detection):
        value = self._embedding(detection)
        # Keep the confirmed reference stable through poor-angle frames.
        if value is not None and (track.embedding is None or not track.recognized):
            track.embedding = value

    def _recover(self, detections, det_ids, track_ids):
        scores = np.full((len(track_ids), len(det_ids)), -1.0)
        for row, t in enumerate(track_ids):
            track = self.tracks[t]
            if track.embedding is None or track.no_losses > 3:
                continue
            box = track.get_bbox()
            center = (box[:2] + box[2:]) / 2
            for col, d in enumerate(det_ids):
                emb = self._embedding(detections[d])
                other = np.asarray(detections[d]['bbox'])
                size = np.maximum(box[2:] - box[:2], other[2:] - other[:2])
                distance = np.linalg.norm(((other[:2] + other[2:]) / 2 - center) / np.maximum(size, 1))
                if emb is not None and emb.shape == track.embedding.shape and distance <= 3:
                    scores[row, col] = float(track.embedding @ emb)
        matches = []
        for row in range(len(track_ids)):
            if not len(det_ids):
                break
            col = int(np.argmax(scores[row]))
            score = scores[row, col]
            alternatives = np.concatenate((np.delete(scores[row], col), np.delete(scores[:, col], row)))
            if score >= 0.65 and (not alternatives.size or score - alternatives.max() >= 0.10):
                matches.append((track_ids[row], det_ids[col]))
        return matches

    def _match(self, det_boxes, trk_boxes):
        iou_matrix = np.zeros((len(trk_boxes), len(det_boxes)))
        for t, tb in enumerate(trk_boxes):
            for d, db in enumerate(det_boxes):
                iou_matrix[t, d] = iou(tb, db)

        t_idx, d_idx  = linear_sum_assignment(-iou_matrix)
        matched_pairs = list(zip(t_idx, d_idx))

        matched     = [(t, d) for t, d in matched_pairs
                       if iou_matrix[t, d] >= self.iou_threshold]
        matched_t   = {t for t, d in matched}
        matched_d   = {d for t, d in matched}
        unmatched_d = [d for d in range(len(det_boxes)) if d not in matched_d]
        unmatched_t = [t for t in range(len(trk_boxes)) if t not in matched_t]

        return matched, unmatched_d, unmatched_t

    def _remove_dead(self):
        self.tracks = [t for t in self.tracks if t.no_losses <= self.max_age]

    def _active(self) -> list:
        return [
            t for t in self.tracks
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits
        ]
