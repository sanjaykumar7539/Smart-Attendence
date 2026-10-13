import threading
import time
import unittest

import cv2
import numpy as np

from camera_reader import CameraReader
from tracker import KalmanFaceTrack
from video_preview import VideoPreview, OverlayTrack, move_overlays


class PreviewTests(unittest.TestCase):
    def test_box_follows_translation_without_changing_identity(self):
        rng = np.random.default_rng(42)
        previous = np.zeros((180,240),np.uint8)
        previous[50:100,60:110] = rng.integers(0,255,(50,50),dtype=np.uint8)
        current = cv2.warpAffine(previous,np.float32([[1,0,9],[0,1,5]]),(240,180))
        track = OverlayTrack((60,50,110,100),'Student','S1',True,'CONFIRMED',0.8,0)
        tracks, people = move_overlays(previous,current,[track],[],(180,240,3))
        self.assertEqual(len(tracks),1)
        np.testing.assert_allclose(tracks[0].bbox,(69,55,119,105),atol=1.5)
        self.assertEqual(tracks[0].employee_id,'S1')
        self.assertEqual(track.bbox,(60,50,110,100))

    def test_lost_image_features_keep_last_box_until_refresh(self):
        gray = np.zeros((100,100),np.uint8)
        track = OverlayTrack((10,10,50,50),'Student','S1',True,'CONFIRMED',0.8,0)
        tracks, _ = move_overlays(gray,gray,[track],[],(100,100,3))
        self.assertEqual(tracks, (track,))

    def test_preview_continues_without_new_inference_and_does_not_consume_ai_frame(self):
        camera = CameraReader()
        frames = []
        preview = VideoPreview(camera, lambda *args: None, frames.append,
                               lambda fps: None, lambda: 0, max_fps=30)
        thread = threading.Thread(target=preview.run)
        thread.start()
        try:
            # No recognition results arrive while the camera keeps capturing.
            for value in range(20):
                with camera._lock:
                    camera._frame = np.full((80, 120, 3), value * 10, np.uint8)
                time.sleep(0.04)
            self.assertGreaterEqual(len(frames), 10)
            self.assertIsNotNone(camera.get_new_frame())
            self.assertIsNone(camera.get_new_frame())
            decoded = cv2.imdecode(np.frombuffer(frames[-1], np.uint8), cv2.IMREAD_COLOR)
            self.assertGreater(float(decoded.mean()), 150)
        finally:
            preview.stop()
            thread.join(timeout=2)

    def test_overlay_snapshot_is_independent_and_drawing_does_not_edit_camera(self):
        camera = CameraReader()
        source = np.zeros((80,120,3),np.uint8)
        camera._frame = source
        seen = []
        ready = threading.Event()
        def draw(frame, tracks, people):
            frame[:] = 255
            seen.extend(tracks)
        preview = VideoPreview(camera, draw, lambda _: ready.set(), lambda _: None, lambda: 0)
        track = KalmanFaceTrack(np.array([1,2,20,30]))
        track.name = 'Original'
        preview.update_overlays([track], [], 0)
        track.name = 'Changed'
        thread = threading.Thread(target=preview.run)
        thread.start()
        try:
            self.assertTrue(ready.wait(2))
            self.assertEqual(seen[0].name, 'Original')
            self.assertEqual(int(source.max()), 0)
        finally:
            preview.stop()
            thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
