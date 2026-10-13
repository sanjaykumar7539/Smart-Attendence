import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from yolov8_face_recognition import YOLOv8FaceRecognitionEngine


class YOLOv8FaceRecognitionTests(unittest.TestCase):
    def make_engine(self, boxes, scores, insight_faces):
        engine = YOLOv8FaceRecognitionEngine.__new__(YOLOv8FaceRecognitionEngine)

        class DetectionBoxes:
            def __init__(self):
                self.xyxy = Mock()
                self.xyxy.cpu.return_value.numpy.return_value = np.asarray(
                    boxes, dtype=np.float32
                ).reshape(-1, 4)
                self.conf = Mock()
                self.conf.cpu.return_value.numpy.return_value = np.asarray(
                    scores, dtype=np.float32
                )

            def __len__(self):
                return len(boxes)

        detection_result = SimpleNamespace(boxes=DetectionBoxes())
        engine.face_model = Mock()
        engine.face_model.predict.return_value = [detection_result]
        engine.app = Mock()
        engine.app.get.return_value = insight_faces
        return engine

    def test_yolov8_face_boxes_gate_insightface_embeddings(self):
        embedding = np.ones(512, dtype=np.float32)
        matched_face = SimpleNamespace(
            bbox=np.asarray([12, 14, 48, 52]),
            kps=np.ones((5, 2), dtype=np.float32),
            normed_embedding=embedding,
        )
        unrelated_face = SimpleNamespace(
            bbox=np.asarray([110, 10, 150, 50]),
            kps=np.ones((5, 2), dtype=np.float32),
            normed_embedding=embedding,
        )
        engine = self.make_engine(
            [[10, 12, 50, 54]], [0.91], [matched_face, unrelated_face]
        )

        faces = engine.detect_faces(np.zeros((100, 180, 3), dtype=np.uint8))

        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0]["bbox"], [10, 12, 50, 54])
        self.assertEqual(faces[0]["confidence"], 0.91)
        np.testing.assert_array_equal(faces[0]["embedding"], embedding)
        engine.app.get.assert_called_once()
        self.assertEqual(engine.last_person_boxes, [])

    def test_empty_yolov8_result_does_not_run_embedding_detection(self):
        engine = self.make_engine([], [], [])

        faces = engine.detect_faces(np.zeros((100, 180, 3), dtype=np.uint8))

        self.assertEqual(faces, [])
        engine.app.get.assert_not_called()

    def test_invalid_frame_is_ignored(self):
        engine = self.make_engine([], [], [])

        self.assertEqual(engine.detect_faces(None), [])
        engine.face_model.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
