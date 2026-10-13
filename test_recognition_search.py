"""Regression checks for CPU and FAISS identity search parity."""
import unittest

import faiss
import numpy as np

from config import config
from recognition import RecognitionEngine


class RecognitionSearchTests(unittest.TestCase):
    def test_cpu_numpy_search_matches_faiss_with_multiple_templates(self):
        vectors = np.asarray(
            [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]],
            dtype=np.float32,
        )
        index = faiss.IndexFlatIP(2)
        index.add(vectors)
        meta = ["student-a", "student-a", "student-b", "student-c"]

        numpy_engine = RecognitionEngine.__new__(RecognitionEngine)
        numpy_engine.index = index
        numpy_engine.meta = meta
        numpy_engine._index_vectors = vectors.copy()

        faiss_engine = RecognitionEngine.__new__(RecognitionEngine)
        faiss_engine.index = index
        faiss_engine.meta = meta
        faiss_engine._index_vectors = None

        old_margin = config.model.min_margin
        old_few_shot = config.model.few_shot_matching
        try:
            config.model.min_margin = 0.0
            config.model.few_shot_matching = False
            numpy_result = numpy_engine.identify_detailed(
                np.asarray([1.0, 0.0], dtype=np.float32), k=4, threshold=0.0
            )
            faiss_result = faiss_engine.identify_detailed(
                np.asarray([1.0, 0.0], dtype=np.float32), k=4, threshold=0.0
            )
        finally:
            config.model.min_margin = old_margin
            config.model.few_shot_matching = old_few_shot

        self.assertEqual(numpy_result[0], "student-a")
        self.assertEqual(faiss_result[0], "student-a")
        np.testing.assert_allclose(numpy_result[1:], faiss_result[1:], atol=1e-6)
        self.assertAlmostEqual(numpy_result[1], 1.0)
        self.assertAlmostEqual(numpy_result[2], 1.0)

    def test_quality_weighted_prototype_and_best_template_are_used(self):
        vectors = np.asarray(
            [[0.8, 0.6], [0.0, 1.0], [1.0, 0.0]],
            dtype=np.float32,
        )
        index = faiss.IndexFlatIP(2)
        index.add(vectors)
        engine = RecognitionEngine.__new__(RecognitionEngine)
        engine.index = index
        engine.meta = ["student-a", "student-a", "student-b"]
        engine._index_vectors = vectors.copy()
        engine._template_weights = np.asarray([1.0, 0.1, 1.0], dtype=np.float32)

        result = engine.identify_detailed(
            np.asarray([0.8, 0.6], dtype=np.float32),
            k=3,
            threshold=0.0,
        )

        self.assertEqual(result[0], "student-a")
        self.assertGreater(result[1], 0.9)
        self.assertGreater(result[2], 0.06)


if __name__ == "__main__":
    unittest.main()
