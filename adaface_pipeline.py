"""Bridge the original AdaFace recognizer to FaceTrack's frame contract."""
from pathlib import Path

import cv2

import adaface_original_core as core
from camera_reader import CameraReader
from super_resolution import RealEsrganFaceEnhancer


SOURCE = Path(__file__).resolve().parent / "adaface_source"


class AdaFaceRecognitionEngine:
    def __init__(self):
        core.ADAFACE_MODEL_DIR = str(SOURCE / "adaface_model")
        core.FACE_MODEL_PATH = str(SOURCE / "yolov8m-face.pt")
        core.FAISS_PATH = str(SOURCE / "data/students.faiss")
        core.METADATA_PATH = str(SOURCE / "data/students.json")
        core.YOLO_IMGSZ = 1280
        self.runtime = core.OriginalRecognitionRuntime()
        enhancer_path = SOURCE.parent / "models" / "RealESRGAN_x2.pth"
        self.runtime.face_enhancer = RealEsrganFaceEnhancer(enhancer_path)

    def process_frame(self, frame, annotate=True):
        if frame is None or frame.size == 0:
            return [], frame
        faces = self.runtime.step(frame)
        # FaceTrack renders the confirmed identities in its existing preview.
        return faces, frame.copy() if annotate else frame

    def reset(self):
        self.runtime.reset()


class AdaFaceCameraReader(CameraReader):
    """Consume recorded video one frame at a time, at its native resolution.

    The regular background reader drops frames when inference is slower than
    playback. The first ZIP processes every frame; retain that behavior here.
    Live camera modes still use FaceTrack's existing camera reader.
    """

    def start(self):
        if self.input_mode != "video" or hasattr(super(), "_loop"):
            return super().start()
        self._capture = cv2.VideoCapture(self.video_path)
        if not self._capture.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

    def get_new_frame(self):
        if self.input_mode != "video":
            return super().get_new_frame()
        if not hasattr(self, "_capture"):
            return super().get_new_frame()
        if self._stop.is_set():
            return None
        with self._lock:
            ok, frame = self._capture.read()
            if not ok:
                return None
            self._frame = frame
            return frame.copy()

    def stop(self):
        super().stop()
        with self._lock:
            if hasattr(self, "_capture"):
                self._capture.release()
