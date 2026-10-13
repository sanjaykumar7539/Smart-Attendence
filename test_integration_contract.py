"""Dependency-free parity checks against the first ZIP's original loop."""
import ast
from collections import defaultdict, deque
from contextlib import redirect_stdout
import io
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent
ORIGINAL = ast.parse((ROOT / "adaface_source/main.py").read_text(encoding="utf-8-sig"))
ADAPTED = ast.parse((ROOT / "adaface_original_core.py").read_text(encoding="utf-8"))


class Tensor:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class Detector:
    def track(self, frame, **kwargs):
        # More tracks than the per-frame budget, a disappearing track, empty
        # detections, a box without an ID, and an invalid rectangle.
        ids = [] if frame == 7 else range(1, 9 if frame < 12 else 7)
        boxes = [SimpleNamespace(id=Tensor(i), xyxy=[Tensor([i, 0, i + 40, 40])]) for i in ids]
        boxes += [SimpleNamespace(id=None), SimpleNamespace(id=Tensor(99), xyxy=[Tensor([0, 0, 0, 0])])]
        return [SimpleNamespace(boxes=boxes), SimpleNamespace(boxes=None)]


def environment():
    env = {"defaultdict": defaultdict, "deque": deque,
           "time": SimpleNamespace(perf_counter=lambda: 0.0)}
    for node in ORIGINAL.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                env[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    env.update(face_detector=Detector(), adaface_model=None, alignment_detector=None,
               faiss_index=None, metadata=[], recognition_cache={},
               vote_history=defaultdict(lambda: deque(maxlen=env["VOTE_WINDOW"])),
               confirmed_tracks={}, round_robin_index=0, frame_number=0,
               total_recognition_attempts=0, successful_embeddings=0,
               attendance_students=set(), attendance=SimpleNamespace(mark_attendance=lambda name: None))
    env["crop_face"] = lambda frame, x1, *box: None if x1 == 8 else (frame, x1)
    env["get_embedding"] = lambda model, aligner, crop: (None, 0.0) if crop[1] == 7 else (crop, 5.0)

    def classify(embedding, *args):
        frame, track = embedding
        name = "student_a" if track % 2 else "student_b"
        status = "UNKNOWN" if track == 6 else "UNCERTAIN" if frame % 4 == 0 else "CONFIRMED"
        return dict(name=name, status=status, score=0.6, second_score=0.2, margin=0.4)

    env["classify_identity"] = classify
    return env


class IntegrationContract(unittest.TestCase):
    def test_numpy_similarity_index_returns_sorted_inner_products(self):
        index_class = next(
            node for node in ADAPTED.body
            if isinstance(node, ast.ClassDef) and node.name == "NumpySimilarityIndex"
        )
        env = {"np": np}
        exec(compile(ast.Module(body=[index_class], type_ignores=[]), "numpy-index", "exec"), env)
        index = env["NumpySimilarityIndex"](
            np.asarray([[1, 0], [0.5, 0.5], [0, 1], [1, 0]], dtype=np.float32)
        )

        scores, ids = index.search(np.asarray([[1, 0]], dtype=np.float32), 3)

        np.testing.assert_array_equal(ids, [[0, 3, 1]])
        np.testing.assert_array_equal(scores, [[1, 1, 0.5]])

    def test_video_adapter_consumes_every_frame_without_resizing(self):
        tree = ast.parse((ROOT / "adaface_pipeline.py").read_text(encoding="utf-8"))
        camera_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AdaFaceCameraReader")

        class Frame:
            def __init__(self, number):
                self.number = number

            def copy(self):
                return Frame(self.number)

        class Capture:
            def __init__(self, path):
                self.frames = iter([Frame(1), Frame(2), Frame(3)])
                self.closed = False

            def isOpened(self):
                return True

            def read(self):
                frame = next(self.frames, None)
                return frame is not None, frame

            def release(self):
                self.closed = True

        class Base:
            def stop(self):
                self._stop.set()

        env = {"CameraReader": Base, "cv2": SimpleNamespace(VideoCapture=Capture)}
        exec(compile(ast.Module(body=[camera_class], type_ignores=[]), "camera-adapter", "exec"), env)
        camera = env["AdaFaceCameraReader"]()
        camera.input_mode, camera.video_path = "video", "test.mp4"
        camera._lock, camera._stop = threading.Lock(), threading.Event()
        camera.start()
        self.assertEqual([camera.get_new_frame().number for _ in range(3)], [1, 2, 3])
        self.assertIsNone(camera.get_new_frame())
        camera.stop()
        self.assertTrue(camera._capture.closed)

    def test_original_functions_and_constants_unchanged(self):
        for kind in (ast.FunctionDef, ast.Assign):
            original = [ast.dump(n) for n in ORIGINAL.body if isinstance(n, kind) and n.lineno < 1065]
            adapted = [ast.dump(n) for n in ADAPTED.body if isinstance(n, kind)]
            self.assertEqual(original, adapted)

    def test_original_loop_matches_adapted_scheduler_votes_and_results(self):
        loop = next(n for n in ORIGINAL.body if isinstance(n, ast.While))
        statements = []
        for node in loop.body[2:]:
            statements.append(node)
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "adaface_ms" for t in node.targets):
                break
        original_step = compile(ast.Module(body=statements, type_ignores=[]), "original-step", "exec")
        original_env, adapted_env = environment(), environment()
        runtime_class = next(
            n for n in ADAPTED.body
            if isinstance(n, ast.ClassDef) and n.name == "OriginalRecognitionRuntime"
        )
        exec(compile(ast.Module(body=[runtime_class], type_ignores=[]), "adapted", "exec"), adapted_env)
        runtime = adapted_env["OriginalRecognitionRuntime"].__new__(adapted_env["OriginalRecognitionRuntime"])
        runtime.__dict__.update(environment())
        runtime.reset()
        for frame in range(1, 31):
            original_env["frame"] = frame
            with redirect_stdout(io.StringIO()):
                exec(original_step, original_env)
            faces = runtime.step(frame)
            for key in ("recognition_cache", "vote_history", "confirmed_tracks", "round_robin_index",
                        "frame_number", "total_recognition_attempts", "successful_embeddings"):
                self.assertEqual(original_env[key], getattr(runtime, key), (frame, key))
            for face in faces:
                identity = original_env["confirmed_tracks"].get(face["source_track_id"])
                self.assertEqual(face["employee_id"], identity)
                self.assertEqual(face["recognized"], identity is not None)
        self.assertTrue(runtime.confirmed_tracks)
        runtime.reset()
        self.assertFalse(runtime.confirmed_tracks)
        self.assertFalse(runtime.vote_history)
        self.assertEqual(runtime.frame_number, 0)

    def test_threshold_boundaries(self):
        env = environment()
        function = next(n for n in ORIGINAL.body if isinstance(n, ast.FunctionDef) and n.name == "classify_identity")
        exec(compile(ast.Module(body=[function], type_ignores=[]), "classification", "exec"), env)
        for area, score, margin, expected in [(3000, .40, .045, "CONFIRMED"),
                                              (2000, .37, .035, "CONFIRMED"),
                                              (3000, .39, .10, "UNCERTAIN"),
                                              (2000, .37, .034, "UNCERTAIN"),
                                              (3000, .31, .10, "UNKNOWN")]:
            env["search_student"] = lambda *args: ("student", score, score - margin, margin)
            self.assertEqual(env["classify_identity"](None, area, None, None)["status"], expected)


if __name__ == "__main__":
    unittest.main()
