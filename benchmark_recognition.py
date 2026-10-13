"""Compare Buffalo_S and Buffalo_L recognition on the same video samples."""
import argparse
import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKS = ("buffalo_s", "buffalo_l")
PACK_FILES = {
    "buffalo_s": ("det_500m.onnx", "w600k_mbf.onnx"),
    "buffalo_l": ("det_10g.onnx", "w600k_r50.onnx"),
}


def gallery_paths(pack):
    directory = ROOT / ("student_index" if pack == "buffalo_s" else f"student_index_{pack}")
    return directory, directory / "students.faiss", directory / "students.json"


def missing_assets(video):
    missing = []
    if not video.is_file():
        missing.append(video)
    if not (ROOT / "adaface_source" / "yolov8m-face.pt").is_file():
        missing.append(ROOT / "adaface_source" / "yolov8m-face.pt")
    for pack in PACKS:
        directory, index_path, metadata_path = gallery_paths(pack)
        for path in (index_path, metadata_path):
            if not path.is_file():
                missing.append(path)
        for filename in PACK_FILES[pack]:
            model_path = ROOT / "models" / "models" / pack / filename
            if not model_path.is_file():
                missing.append(model_path)
    return missing


def evaluate_pack(pack, video, max_frames, stride):
    import cv2
    import faiss
    import numpy as np

    from config import config
    from quality import FaceQualityAnalyzer
    from yolov8_face_recognition import YOLOv8FaceRecognitionEngine

    gallery_dir, index_path, source_meta = gallery_paths(pack)
    rows = json.loads(source_meta.read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_path))
    if not rows or index.ntotal != len(rows) or index.d != 512:
        raise RuntimeError(f"{pack} gallery is empty or inconsistent: {gallery_dir}")
    if any(not row.get("student_id") for row in rows):
        raise RuntimeError(f"{pack} gallery metadata is missing student IDs")

    with tempfile.TemporaryDirectory(prefix=f"facetrack-{pack}-") as temporary:
        runtime_meta = Path(temporary) / "meta.json"
        runtime_meta.write_text(
            json.dumps([row["student_id"] for row in rows], ensure_ascii=False),
            encoding="utf-8",
        )
        config.database.index_path = str(index_path)
        config.database.meta_path = str(runtime_meta)
        config.model.detector = "YOLOv8 Face + InsightFace"
        config.model.model_pack = pack
        config.model.models_dir = str(ROOT / "models")
        config.model.det_size = (640, 640)
        config.model.use_gpu = False

        engine = YOLOv8FaceRecognitionEngine()
        quality_analyzer = FaceQualityAnalyzer(config.model)
        thresholds = {
            "good": config.model.threshold_good,
            "medium": config.model.threshold_medium,
            "poor": config.model.threshold_poor,
        }
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open benchmark video: {video}")

        sample_number = 0
        frame_number = 0
        face_detections = 0
        accepted_faces = 0
        recognized_students = set()
        try:
            while sample_number < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                current_frame = frame_number
                frame_number += 1
                if current_frame % stride:
                    continue
                if frame.shape[1] > 1280:
                    frame = cv2.resize(frame, (1280, 720))

                sample_number += 1
                faces = engine.detect_faces(frame)
                face_detections += len(faces)
                for face in faces:
                    quality = quality_analyzer.analyze(
                        frame, face["bbox"], face.get("landmark")
                    )
                    identity, _score, _margin = engine.identify_detailed(
                        face["embedding"],
                        k=config.model.max_search_k,
                        threshold=thresholds[quality.level],
                    )
                    if identity:
                        accepted_faces += 1
                        recognized_students.add(identity)
        finally:
            capture.release()

    return {
        "model_pack": pack,
        "sampled_frames": sample_number,
        "face_detections": face_detections,
        "accepted_face_matches": accepted_faces,
        "unique_accepted_students": len(recognized_students),
        "accepted_student_ids": sorted(recognized_students),
        "thresholds": thresholds,
        "minimum_margin": config.model.min_margin,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "D10_20261009102035_1.mp4")
    parser.add_argument("--frames", type=int, default=12, help="Maximum sampled frames per model")
    parser.add_argument("--stride", type=int, default=25, help="Sample every Nth video frame")
    parser.add_argument("--check", action="store_true", help="Check both models and galleries without loading them")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    video = args.video.expanduser().resolve()
    if args.frames < 1 or args.stride < 1:
        parser.error("--frames and --stride must be positive")

    missing = missing_assets(video)
    if missing:
        print("Benchmark cannot run; missing local assets:")
        print("\n".join(str(path) for path in missing))
        if any("buffalo_l" in str(path) for path in missing):
            print(
                "Install the Buffalo_L InsightFace pack and build its separate "
                "gallery with enroll_csbs_students.py --model-pack buffalo_l."
            )
        raise SystemExit(1)
    if args.check:
        print("Both model packs, galleries, and the video are present.")
        return

    os.chdir(ROOT)
    results = [
        evaluate_pack(pack, video, args.frames, args.stride)
        for pack in PACKS
    ]
    result = {
        "video": str(video),
        "sampling": {"max_frames_per_model": args.frames, "stride": args.stride},
        "results": results,
        "note": (
            "Counts are accepted face matches, not verified unique people or "
            "attendance accuracy. Review IDs and test against labeled frames "
            "before changing the production model."
        ),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Benchmark report saved: {output}")


if __name__ == "__main__":
    main()
