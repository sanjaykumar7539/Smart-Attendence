"""Tests for model-specific recognition benchmark asset isolation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark_recognition


class RecognitionBenchmarkTests(unittest.TestCase):
    def test_model_packs_use_separate_galleries(self):
        small, small_index, _ = benchmark_recognition.gallery_paths("buffalo_s")
        large, large_index, _ = benchmark_recognition.gallery_paths("buffalo_l")
        self.assertNotEqual(small, large)
        self.assertNotEqual(small_index, large_index)

    def test_missing_local_weights_are_reported_without_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "classroom.mp4"
            video.touch()
            detector = root / "adaface_source" / "yolov8m-face.pt"
            detector.parent.mkdir()
            detector.touch()
            with patch.object(benchmark_recognition, "ROOT", root):
                for pack in benchmark_recognition.PACKS:
                    gallery, index, metadata = benchmark_recognition.gallery_paths(pack)
                    gallery.mkdir(parents=True)
                    index.touch()
                    metadata.touch()
                    model_directory = root / "models" / "models" / pack
                    model_directory.mkdir(parents=True)
                    for filename in benchmark_recognition.PACK_FILES[pack]:
                        if pack == "buffalo_s":
                            (model_directory / filename).touch()
                missing = benchmark_recognition.missing_assets(video)

            self.assertEqual(
                missing,
                [
                    root / "models" / "models" / "buffalo_l" / "det_10g.onnx",
                    root / "models" / "models" / "buffalo_l" / "w600k_r50.onnx",
                ],
            )


if __name__ == "__main__":
    unittest.main()
