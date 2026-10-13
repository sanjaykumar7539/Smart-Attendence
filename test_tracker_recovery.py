import unittest
import numpy as np
from tracker import SORTTracker


def face(x, embedding):
    return {'bbox': [x, 0, x + 20, 20], 'embedding': embedding}


class RecoveryTests(unittest.TestCase):
    def tracker(self):
        tracker = SORTTracker(min_hits=1)
        tracker.update([face(0, [1, 0])])
        track = tracker.tracks[0]
        track.employee_id = 'A'
        track.name = 'Student A'
        track.recognized = True
        track.recognition_status = 'CONFIRMED'
        return tracker, track

    def test_motion_gap_preserves_confirmed_identity(self):
        tracker, track = self.tracker()
        tracker.update([])
        tracker.update([face(25, [1, 0])])
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(track.no_losses, 0)
        self.assertEqual(track.employee_id, 'A')
        self.assertTrue(track.recognized)

    def test_other_face_does_not_inherit_identity(self):
        tracker, track = self.tracker()
        tracker.update([face(25, [0, 1])])
        self.assertEqual(len(tracker.tracks), 2)
        self.assertFalse(tracker.tracks[-1].recognized)

    def test_ambiguous_faces_are_not_recovered(self):
        tracker, track = self.tracker()
        tracker.update([face(25, [1, 0]), face(45, [1, 0])])
        self.assertGreater(track.no_losses, 0)

    def test_long_gap_or_distant_face_is_not_recovered(self):
        for gap, x in [(4, 25), (0, 200)]:
            tracker, track = self.tracker()
            for _ in range(gap):
                tracker.update([])
            tracker.update([face(x, [1, 0])])
            self.assertEqual(len(tracker.tracks), 2)

    def test_bbox_only_callers_still_work(self):
        tracker = SORTTracker(min_hits=1)
        tracker.update([{'bbox': [0, 0, 20, 20]}])
        tracker.update([{'bbox': [1, 0, 21, 20]}])
        self.assertEqual(len(tracker.tracks), 1)


if __name__ == '__main__':
    unittest.main()
