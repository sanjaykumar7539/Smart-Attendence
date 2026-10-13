"""Regression checks for classroom presence without line crossing."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from config import config


class PresenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.old_path = config.database.db_path
        config.database.db_path = str(Path(cls.temp.name) / 'attendance.db')
        from main import AttendanceTrigger, _draw_overlays
        from database import db
        cls.trigger_type = AttendanceTrigger
        cls.draw = staticmethod(_draw_overlays)
        cls.db = db

    @classmethod
    def tearDownClass(cls):
        import gc
        config.database.db_path = cls.old_path
        gc.collect()  # Release SQLite context-manager connections before Windows cleanup.
        cls.temp.cleanup()

    def setUp(self):
        self.old_mode = config.tracker.attendance_mode
        self.old_roster_ids = config.database.student_roster_ids
        config.tracker.attendance_mode = 'presence'
        config.database.student_roster_ids = None

    def tearDown(self):
        config.tracker.attendance_mode = self.old_mode
        config.database.student_roster_ids = self.old_roster_ids

    def track(self, identity, y):
        from tracker import KalmanFaceTrack
        t = KalmanFaceTrack(np.array([20, y, 70, y + 50]))
        t.employee_id, t.recognized = identity, True
        return t

    def test_stationary_faces_anywhere_and_duplicate_identity(self):
        trigger = self.trigger_type()
        self.assertTrue(trigger.check(self.track('A', 0), 720))
        self.assertTrue(trigger.check(self.track('B', 650), 720))
        self.assertFalse(trigger.check(self.track('A', 300), 720))

    def test_presence_allows_same_student_again_on_a_new_day(self):
        from datetime import date, timedelta
        today = date.today()
        trigger = self.trigger_type()
        with patch('main.datetime') as clock:
            clock.now.return_value.date.side_effect = [today, today, today + timedelta(days=1)]
            self.assertTrue(trigger.check(self.track('A', 0), 720))
            self.assertFalse(trigger.check(self.track('A', 0), 720))
            self.assertTrue(trigger.check(self.track('A', 0), 720))

    def test_unknown_and_predicted_only_faces_not_marked(self):
        trigger = self.trigger_type()
        t = self.track('A', 10)
        t.no_losses = 1
        self.assertFalse(trigger.check(t, 720))
        t.no_losses, t.recognized = 0, False
        self.assertFalse(trigger.check(t, 720))

    def test_presence_bypasses_schedule_and_keeps_one_daily_row(self):
        self.db.add_faculty('A', 'Student A', staff_category='Student')
        with patch.object(self.db, '_update_attendance', side_effect=AssertionError('schedule used')):
            self.db.log_presence('A', 'TEST', 0.9)
            self.db.log_presence('A', 'TEST', 0.9)
        with self.db._connect() as conn:
            rows = conn.execute('SELECT final_status FROM attendance WHERE employee_id=?', ('A',)).fetchall()
            self.assertEqual([r[0] for r in rows], ['PRESENT'])
            self.assertEqual(conn.execute('SELECT count(*) FROM monthly_permissions').fetchone()[0], 0)

    def test_no_detection_line_overlay(self):
        with patch('main.cv2.line') as line:
            self.draw(np.zeros((720, 1280, 3), np.uint8), [], 720, 1280)
            line.assert_not_called()

    def test_presence_hud_hides_unenrolled_detection_totals(self):
        from main import _draw_hud
        with patch('main.cv2.putText') as put_text:
            _draw_hud(np.zeros((720, 1280, 3), np.uint8), 1280, 720)
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertFalse(any(label.startswith('Total:') for label in labels))

    def test_capture_only_consumed_once(self):
        from camera_reader import CameraReader
        camera = CameraReader()
        camera._frame = np.zeros((10, 10, 3), np.uint8)
        self.assertIsNotNone(camera.get_new_frame())
        self.assertIsNone(camera.get_new_frame())
        camera._frame = camera._frame.copy()
        self.assertIsNotNone(camera.get_new_frame())

    def test_green_only_after_confirmation_and_person_dedup(self):
        from main import _confirmed, _unmatched_people
        track = self.track('A', 100)
        self.assertFalse(_confirmed(track))
        track.recognition_status = 'CONFIRMED'
        self.assertTrue(_confirmed(track))
        people = [[0, 80, 100, 300], [200, 80, 300, 300]]
        self.assertEqual(_unmatched_people([track], people), [people[1]])
        overlapping = [[0, 80, 100, 300], [0, 80, 200, 400]]
        self.assertEqual(_unmatched_people([track], overlapping), [overlapping[1]])
        frame = np.zeros((720, 1280, 3), np.uint8)
        self.draw(frame, [track], 720, 1280)
        np.testing.assert_array_equal(frame[100, 45], (0, 255, 0))
        track.recognition_status = 'UNKNOWN'
        frame = np.zeros((720, 1280, 3), np.uint8)
        self.draw(frame, [track], 720, 1280)
        np.testing.assert_array_equal(frame[100, 45], (0, 0, 255))

    def test_overlay_hides_unrecognized_labels_but_keeps_confirmed_name(self):
        from main import _draw_overlays
        unknown = self.track('A', 100)
        frame = np.zeros((720, 1280, 3), np.uint8)
        with patch('main.cv2.putText') as put_text:
            _draw_overlays(frame, [unknown], 720, 1280, [[200, 100, 260, 180]])
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertNotIn('Unknown', labels)
        self.assertNotIn('Not confirmed', labels)
        np.testing.assert_array_equal(frame[100, 45], (0, 0, 255))

        recognized = self.track('A', 100)
        recognized.name = 'Student A'
        recognized.confidence = 0.91
        recognized.recognition_status = 'CONFIRMED'
        frame = np.zeros((720, 1280, 3), np.uint8)
        with patch('main.cv2.putText') as put_text:
            _draw_overlays(frame, [recognized], 720, 1280)
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn('Student A', labels)
        self.assertIn('91%', labels)

    def test_reset_scope_backup_and_enrollment_preserved(self):
        from datetime import date, timedelta
        from database import AttendanceDB
        import sqlite3
        path = Path(self.temp.name) / 'reset' / 'test.db'
        database = AttendanceDB(str(path))
        database.add_faculty('RESET_TEST', 'Reset Test', staff_category='Student')
        database.log_presence('RESET_TEST', 'TEST', 0.9)
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        with database._connect() as conn:
            conn.execute("INSERT INTO attendance(employee_id,work_date,final_status) VALUES(?,?,'PRESENT')", ('RESET_TEST',yesterday))
            conn.execute("INSERT INTO detections(employee_id,camera_id,direction,work_date) VALUES(?,'TEST','IN',?)", ('RESET_TEST',yesterday))
        result = database.reset_presence('previous')
        self.assertEqual(result['attendance_deleted'], 1)
        self.assertEqual(result['detections_deleted'], 1)
        self.assertEqual(len(database.get_today_attendance()), 1)
        backup = next((path.parent / 'backups').glob('*.db'))
        conn = sqlite3.connect(backup)
        self.assertEqual(conn.execute('SELECT count(*) FROM attendance').fetchone()[0], 2)
        conn.close()
        database.reset_presence('today')
        self.assertEqual(database.get_today_attendance(), [])
        self.assertIsNotNone(database.get_faculty('RESET_TEST'))

    def test_startup_today_reset_preserves_older_history_and_backs_up_current_data(self):
        from datetime import date, timedelta
        from database import AttendanceDB
        import sqlite3

        path = Path(self.temp.name) / 'startup_reset' / 'test.db'
        database = AttendanceDB(str(path))
        database.add_faculty('STARTUP_TEST', 'Startup Test', staff_category='Student')
        database.log_presence('STARTUP_TEST', 'TEST', 0.9)
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        with database._connect() as conn:
            conn.execute(
                "INSERT INTO attendance(employee_id,work_date,final_status) VALUES(?,?,'PRESENT')",
                ('STARTUP_TEST', yesterday),
            )
            conn.execute(
                "INSERT INTO detections(employee_id,camera_id,direction,work_date) VALUES(?,'TEST','IN',?)",
                ('STARTUP_TEST', yesterday),
            )

        result = database.reset_presence('today')

        self.assertEqual(result['attendance_deleted'], 1)
        self.assertEqual(result['detections_deleted'], 1)
        with database._connect() as conn:
            self.assertEqual(
                conn.execute('SELECT work_date FROM attendance').fetchall()[0][0],
                yesterday,
            )
            self.assertEqual(
                conn.execute('SELECT work_date FROM detections').fetchall()[0][0],
                yesterday,
            )
        self.assertIsNotNone(database.get_faculty('STARTUP_TEST'))

        backups = list((path.parent / 'backups').glob('*.db'))
        self.assertEqual(len(backups), 1)
        with sqlite3.connect(backups[0]) as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM attendance').fetchone()[0], 2)
            self.assertEqual(conn.execute('SELECT count(*) FROM detections').fetchone()[0], 2)
            self.assertEqual(conn.execute('SELECT count(*) FROM faculty').fetchone()[0], 1)

    def test_current_day_roster_and_event_filter(self):
        import asyncio
        from datetime import date
        from main import get_attendance, _today_events, state
        self.db.add_faculty('UNSEEN', 'Unseen Student', staff_category='Student')
        rows = asyncio.run(get_attendance())
        unseen = next(r for r in rows if r['student_id'] == 'UNSEEN')
        self.assertEqual(unseen['final_status'], 'ABSENT')
        self.assertEqual(unseen['work_date'], date.today().isoformat())
        original = state.latest_events
        try:
            current = {'work_date':date.today().isoformat(), 'generation':state.reset_generation}
            state.latest_events = [current, {'work_date':'2000-01-01', 'generation':state.reset_generation}]
            self.assertEqual(_today_events(), [current])
        finally:
            state.latest_events = original

    def test_student_dashboard_uses_only_enrolled_roster_data(self):
        import asyncio
        from main import _dashboard_stats, _today_events, dashboard, get_attendance, state
        from datetime import datetime
        db = self.db
        db.add_faculty('ROSTER_A', 'Roster Student A', staff_category='Student')
        db.add_faculty('ROSTER_B', 'Roster Student B', staff_category='Student')
        db.add_faculty('STALE_STAFF', 'Unrelated Staff', staff_category='Teaching')
        db.log_presence('ROSTER_A', 'TEST', 0.9)
        config.database.student_roster_ids = ['ROSTER_A', 'ROSTER_B', 'ROSTER_A']

        attendance = asyncio.run(get_attendance())
        self.assertEqual([row['employee_id'] for row in attendance], ['ROSTER_A', 'ROSTER_B'])
        page = asyncio.run(dashboard())
        self.assertIn('<body class="student-mode">', page)
        stats = _dashboard_stats()
        self.assertEqual((stats['total'], stats['present'], stats['absent']), (2, 1, 1))

        original_events = state.latest_events
        try:
            today = datetime.now().date().isoformat()
            generation = state.reset_generation
            state.latest_events = [
                {'employee_id': 'ROSTER_A', 'work_date': today, 'generation': generation},
                {'employee_id': 'STALE_STAFF', 'work_date': today, 'generation': generation},
            ]
            self.assertEqual(
                [event['employee_id'] for event in _today_events()],
                ['ROSTER_A'],
            )
        finally:
            state.latest_events = original_events


if __name__ == '__main__':
    unittest.main()
