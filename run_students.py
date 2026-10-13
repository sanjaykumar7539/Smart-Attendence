"""Enroll dataset photos and run the video dashboard with a separate student DB.

Usage: python run_students.py --enroll   (rebuild gallery, then run)
       python run_students.py            (reuse gallery, enroll if missing)
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def dataset_signature():
    entries = []
    for folder in sorted((ROOT / 'dataset').iterdir()):
        if not folder.is_dir() or folder.name.upper() == 'MIXED':
            continue
        for photo in sorted(folder.iterdir()):
            if photo.is_file() and photo.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                stat = photo.stat()
                entries.append((str(photo.relative_to(ROOT)), stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enroll', action='store_true')
    parser.add_argument('--enroll-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)

    from config import config

    # Avoid starting a second model/camera when the dashboard is already running.
    if not args.enroll_only:
        import socket
        import webbrowser
        port = config.dashboard.port
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', port)) == 0:
                print(f'The app may already be running on port {port}. Stop its terminal with Ctrl+C and reopen run_students.bat to apply config.py changes. Opening the configured dashboard URL.', flush=True)
                webbrowser.open(f'http://localhost:{port}')
                return

    if config.camera.input_mode == 'video' and not args.enroll_only:
        video_path = Path(config.camera.video_path).expanduser()
        if not video_path.is_absolute():
            video_path = ROOT / video_path
        video_path = video_path.resolve()
        if not video_path.is_file():
            raise SystemExit(f'Video not found: {video_path}\nSet CameraConfig.video_path in config.py to an existing video file.')
        config.camera.video_path = str(video_path)
        print(f'Video from config.py: {video_path}', flush=True)
        import cv2
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise SystemExit(f'Unable to read video dimensions: {video_path}')
            config.camera.frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            config.camera.frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        if config.camera.frame_width <= 0 or config.camera.frame_height <= 0:
            raise SystemExit(f'Video has invalid dimensions: {video_path}')
    gallery = ROOT / 'student_index'
    index_path = gallery / 'students.faiss'
    source_meta = gallery / 'students.json'
    signature_path = gallery / 'dataset_signature.txt'
    signature = dataset_signature()
    changed = not signature_path.exists() or signature_path.read_text() != signature
    if args.enroll or args.enroll_only or changed or not (index_path.exists() and source_meta.exists()):
        from enroll_csbs_students import main as enroll
        enroll()
        signature_path.write_text(signature)

    import faiss
    rows = json.loads(source_meta.read_text(encoding='utf-8'))
    index = faiss.read_index(str(index_path))
    if not rows or index.ntotal != len(rows) or index.d != 512:
        raise RuntimeError('Student gallery is empty or inconsistent; rerun with --enroll')
    identities = [row['student_id'] for row in rows]
    config.database.student_roster_ids = list(dict.fromkeys(identities))
    runtime_meta = gallery / 'runtime_meta.json'
    runtime_meta.write_text(json.dumps(identities, ensure_ascii=False, indent=2), encoding='utf-8')
    quality_weights = [float(row.get('quality_weight', 1.0)) for row in rows]
    runtime_meta.with_suffix('.quality.json').write_text(
        json.dumps(quality_weights, indent=2),
        encoding='utf-8',
    )

    # Configure before importing database/main, which create their objects on import.
    config.database.db_path = str(gallery / 'student_attendance.db')
    config.database.index_path = str(index_path)
    config.database.meta_path = str(runtime_meta)
    config.camera.location = 'Student Classroom'
    config.hlc.enabled = False
    config.ai.enabled = True
    config.tracker.attendance_mode = 'presence'
    config.model.detector = 'YOLOv8 Face + InsightFace'
    config.model.det_size = (640, 640)
    from database import db
    seen = set()
    for position, row in enumerate(rows):
        identity = row['student_id']
        if identity in seen:
            continue
        db.add_faculty(identity, row['student_name'], designation='Student', staff_category='Student')
        db.set_embedding_index(identity, position)
        seen.add(identity)
    print(f'Student gallery ready: {len(seen)} students, {index.ntotal} embeddings', flush=True)
    if not args.enroll_only:
        reset = db.reset_presence('today')
        print(
            'Fresh dashboard: cleared '
            f"{reset['attendance_deleted']} attendance rows and "
            f"{reset['detections_deleted']} detections from today; "
            f"backup_saved={reset['backup_saved']}",
            flush=True,
        )
        from main import main as run
        run()


if __name__ == '__main__':
    main()
