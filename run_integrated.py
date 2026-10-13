"""Start the FaceTrack dashboard using the enrolled classroom student gallery."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "adaface_source"
PACKAGES = {
    "cv2": "opencv-python", "numpy": "numpy", "torch": "torch",
    "torchvision": "torchvision", "ultralytics": "ultralytics",
    "insightface": "insightface", "onnxruntime": "onnxruntime",
    "faiss": "faiss-cpu",
    "fastapi": "fastapi", "uvicorn": "uvicorn", "filterpy": "filterpy",
    "scipy": "scipy", "loguru": "loguru", "dotenv": "python-dotenv",
    "httpx": "httpx", "multipart": "python-multipart",
    "yaml": "PyYAML",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "D10_20261009102035_1.mp4")
    parser.add_argument(
        "--model-pack",
        choices=("buffalo_s", "buffalo_l"),
        default="buffalo_s",
        help="Pack matching the selected pack-specific enrollment gallery",
    )
    parser.add_argument("--check", action="store_true", help="Check assets/dependencies without starting models or writing attendance")
    args = parser.parse_args()
    video = args.video.expanduser().resolve()
    student_index = ROOT / (
        "student_index" if args.model_pack == "buffalo_s"
        else f"student_index_{args.model_pack}"
    )
    model_files = (
        ["det_500m.onnx", "w600k_mbf.onnx"]
        if args.model_pack == "buffalo_s"
        else ["det_10g.onnx", "w600k_r50.onnx"]
    )
    required = [
        video,
        SOURCE / "yolov8m-face.pt",
        student_index / "students.faiss",
        student_index / "students.json",
        *(ROOT / "models" / "models" / args.model_pack / name for name in model_files),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    packages = [package for module, package in PACKAGES.items() if importlib.util.find_spec(module) is None]
    if missing or packages:
        if missing:
            print("Missing files:\n" + "\n".join(missing))
        if packages:
            print("Missing Python packages: " + ", ".join(packages))
            print("Use an environment with both projects' dependencies; see INTEGRATION.md.")
        raise SystemExit(1)
    if args.check:
        print("Required assets and Python modules are present. Model inference has not been tested.")
        return

    os.chdir(ROOT)
    import cv2
    import faiss
    from config import config

    index = faiss.read_index(str(student_index / "students.faiss"))
    rows = json.loads((student_index / "students.json").read_text(encoding="utf-8"))
    if not rows or index.ntotal != len(rows) or index.d != 512:
        raise RuntimeError("Enrolled student gallery is empty or inconsistent")
    if any(not row.get("student_id") or not row.get("student_name") for row in rows):
        raise RuntimeError("Enrolled student metadata is missing an ID or name")
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError("Video has invalid dimensions")

    runtime = ROOT / "integrated_data"
    runtime.mkdir(exist_ok=True)
    identities = [row["student_id"] for row in rows]
    config.database.student_roster_ids = list(dict.fromkeys(identities))
    meta = runtime / f"runtime_meta_{args.model_pack}.json"
    meta.write_text(json.dumps(identities, ensure_ascii=False, indent=2), encoding="utf-8")
    database_name = (
        "student_attendance.db" if args.model_pack == "buffalo_s"
        else f"student_attendance_{args.model_pack}.db"
    )
    config.database.db_path = str(runtime / database_name)
    config.database.index_path = str(student_index / "students.faiss")
    config.database.meta_path = str(meta)
    config.camera.input_mode = "video"
    config.camera.use_webcam = False
    config.camera.video_path = str(video)
    config.camera.video_loop = False
    config.camera.frame_width = width
    config.camera.frame_height = height
    config.camera.frame_skip = 1
    config.camera.location = "Student Classroom"
    config.hlc.enabled = False
    config.ai.enabled = True
    config.tracker.attendance_mode = "presence"
    config.model.detector = "YOLOv8 Face + InsightFace"
    config.model.model_pack = args.model_pack
    config.model.models_dir = str(ROOT / "models")
    config.model.det_size = (640, 640)
    config.model.use_gpu = False

    from database import db
    seen = set()
    for position, identity in enumerate(identities):
        if identity not in seen:
            row = rows[position]
            db.add_faculty(
                identity,
                row["student_name"],
                designation="Student",
                staff_category="Student",
            )
            db.set_embedding_index(identity, position)
            seen.add(identity)
    print(f"Verified student gallery: {len(seen)} students, {index.ntotal} embeddings")
    reset = db.reset_presence("today")
    print(
        "Fresh dashboard: cleared "
        f"{reset['attendance_deleted']} attendance rows and "
        f"{reset['detections_deleted']} detections from today; "
        f"backup_saved={reset['backup_saved']}",
        flush=True,
    )
    from main import main as run
    run()


if __name__ == "__main__":
    main()
