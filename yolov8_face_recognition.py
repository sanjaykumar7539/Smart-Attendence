"""YOLOv8 face detection with InsightFace embeddings and FAISS identity search."""
from pathlib import Path

import numpy as np
from loguru import logger

from config import config
from recognition import RecognitionEngine


class YOLOv8FaceRecognitionEngine(RecognitionEngine):
    def __init__(self):
        model_path = Path(config.model.yolo_face_model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing YOLOv8 face weights: {model_path}")

        from ultralytics import YOLO

        self.face_model = YOLO(str(model_path))
        self.last_person_boxes = []
        super().__init__()
        logger.success(
            "YOLOv8 face detection + InsightFace/ArcFace recognition ready"
        )

    @staticmethod
    def _overlap_ratio(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        return intersection / max(min(area_a, area_b), 1e-6)

    def detect_faces(self, frame):
        self.last_person_boxes = []
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            return []

        result = self.face_model.predict(
            source=frame,
            imgsz=config.model.yolo_face_image_size,
            conf=config.model.yolo_face_confidence,
            iou=config.model.yolo_face_iou,
            device=config.model.gpu_id if config.model.use_gpu else "cpu",
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        height, width = frame.shape[:2]
        detected_boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        candidates = []
        for coordinates, score in zip(detected_boxes, scores):
            x1, y1, x2, y2 = map(float, coordinates)
            box = [
                max(0.0, min(x1, width - 1.0)),
                max(0.0, min(y1, height - 1.0)),
                max(0.0, min(x2, width - 1.0)),
                max(0.0, min(y2, height - 1.0)),
            ]
            if box[2] > box[0] and box[3] > box[1]:
                candidates.append((box, round(float(score), 6)))

        if not candidates:
            return []

        insight_faces = self.app.get(frame)
        matched = set()
        output = []
        for box, score in sorted(candidates, key=lambda candidate: -candidate[1]):
            best_index = None
            best_overlap = 0.2
            for index, face in enumerate(insight_faces):
                if index in matched:
                    continue
                overlap = self._overlap_ratio(box, face.bbox)
                if overlap > best_overlap:
                    best_index = index
                    best_overlap = overlap
            if best_index is None:
                continue

            face = insight_faces[best_index]
            embedding = face.normed_embedding
            if embedding is None:
                continue
            matched.add(best_index)
            landmarks = getattr(face, "kps", None)
            output.append({
                "bbox": [int(round(value)) for value in box],
                "landmark": landmarks,
                "embedding": np.asarray(embedding, dtype=np.float32),
                "confidence": score,
            })

        return output
