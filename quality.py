"""Face quality scoring for adaptive recognition."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class FaceQuality:
    score: float
    level: str
    min_side: int
    blur: float
    brightness: float
    pose_score: float


class FaceQualityAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg

    def analyze(self, frame: np.ndarray, bbox, landmarks=None) -> FaceQuality:
        h, w = frame.shape[:2]
        x1,y1,x2,y2 = [int(v) for v in bbox]
        x1,y1=max(0,x1),max(0,y1); x2,y2=min(w,x2),min(h,y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return FaceQuality(0.0, 'poor', 0, 0.0, 0.0, 0.0)
        side = max(1, min(x2-x1, y2-y1))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        bright = float(gray.mean())

        size_s = min(1.0, side / max(1, self.cfg.good_face_px))
        blur_s = np.clip((blur-self.cfg.blur_min)/max(1.0,self.cfg.blur_good-self.cfg.blur_min),0,1)
        if self.cfg.brightness_min <= bright <= self.cfg.brightness_max:
            light_s = 1.0
        else:
            d = min(abs(bright-self.cfg.brightness_min), abs(bright-self.cfg.brightness_max))
            light_s = float(max(0.0, 1.0-d/80.0))
        pose_s = self._pose_score(landmarks)
        score = float(0.40*size_s + 0.30*blur_s + 0.20*light_s + 0.10*pose_s)
        if side >= self.cfg.good_face_px and blur >= self.cfg.blur_good and light_s >= .75:
            level='good'
        elif side >= self.cfg.medium_face_px and blur >= self.cfg.blur_min:
            level='medium'
        else:
            level='poor'
        return FaceQuality(score, level, side, blur, bright, pose_s)

    @staticmethod
    def _pose_score(kps):
        if kps is None or len(kps) < 5: return 0.7
        k=np.asarray(kps,dtype=np.float32)
        le,re,nose=k[0],k[1],k[2]
        eye_mid=(le+re)/2.0
        half=max(1.0,np.linalg.norm(re-le)/2.0)
        asym=np.linalg.norm(nose-eye_mid)/half
        return float(np.clip(1.0-asym/1.25,0,1))


def enrollment_quality_weight(image: np.ndarray, face) -> float:
    """Return a bounded image-quality weight for a valid enrollment embedding."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = np.asarray(face.bbox, dtype=np.int32)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.1

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    size_score = min(1.0, min(crop.shape[:2]) / 160.0)
    blur_score = float(np.clip((blur - 25.0) / 125.0, 0.0, 1.0))
    light_score = float(np.clip(1.0 - abs(brightness - 128.0) / 128.0, 0.0, 1.0))
    pose_score = FaceQualityAnalyzer._pose_score(getattr(face, "kps", None))
    return float(np.clip(
        0.40 * size_score + 0.30 * blur_score +
        0.20 * light_score + 0.10 * pose_score,
        0.10, 1.0,
    ))
