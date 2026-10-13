import time
import json
from pathlib import Path

import cv2
import onnxruntime as ort
from insightface.app import FaceAnalysis

PROJECT_ROOT = Path.home() / "facetrack" / "facetrack_project_complete"
VIDEO_ROOT = Path.home() / "facetrack" / "phase5_validation"
RESULTS_DIR = PROJECT_ROOT / "student_index"
RESULTS_PATH = RESULTS_DIR / "phase5_2_frame_sampling_results.json"
MODEL_ROOT = PROJECT_ROOT / "models"

MODEL_NAME = "buffalo_s"
DET_SIZE = (640, 640)
SAMPLING_RATES = [1, 2, 5, 10]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}


def banner(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def sample_interval(source_fps, target_fps):
    return max(1, int(round(source_fps / target_fps)))


def main():
    overall_start = time.perf_counter()

    banner("FACETRACK — PHASE 5.2 GPU FRAME SAMPLING BENCHMARK")

    print("Video root :", VIDEO_ROOT)
    print("Model root :", MODEL_ROOT)
    print("Model      :", MODEL_NAME)
    print("Det size   :", DET_SIZE)
    print("Rates      :", SAMPLING_RATES)

    banner("ONNX RUNTIME PROVIDER VALIDATION")

    providers = ort.get_available_providers()

    for provider in providers:
        print("  ", provider)

    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError("CUDAExecutionProvider is unavailable.")

    print()
    print("CUDAExecutionProvider: AVAILABLE")

    videos = sorted(
        p for p in VIDEO_ROOT.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        raise RuntimeError("No validation videos found.")

    print()
    print("Validation videos:", len(videos))

    for video in videos:
        print("  ", video.name)

    banner("LOADING INSIGHTFACE")

    model_start = time.perf_counter()

    app = FaceAnalysis(
        name=MODEL_NAME,
        root=str(MODEL_ROOT),
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    app.prepare(ctx_id=0, det_size=DET_SIZE)

    model_load_time = time.perf_counter() - model_start

    print("Model load time: %.4f sec" % model_load_time)

    results = {
        "phase": "5.2",
        "description": "GPU frame sampling benchmark",
        "model": MODEL_NAME,
        "det_size": list(DET_SIZE),
        "providers": providers,
        "sampling_rates_fps": SAMPLING_RATES,
        "videos": [],
    }

    for video_path in videos:
        banner("VIDEO: %s" % video_path.name)

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print("ERROR: Could not open video.")
            continue

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / source_fps if source_fps else 0

        print("Source FPS   : %.2f" % source_fps)
        print("Total frames : %d" % total_frames)
        print("Duration     : %.2f sec" % duration)

        video_result = {
            "video": video_path.name,
            "source_fps": source_fps,
            "total_frames": total_frames,
            "duration_sec": duration,
            "rates": [],
        }

        for target_fps in SAMPLING_RATES:
            interval = sample_interval(source_fps, target_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            sampled_frames = 0
            detected_faces = 0
            decode_time = 0.0
            inference_time = 0.0

            benchmark_start = time.perf_counter()
            frame_number = 0

            while True:
                decode_start = time.perf_counter()
                ok, frame = cap.read()
                decode_time += time.perf_counter() - decode_start

                if not ok:
                    break

                if frame_number % interval != 0:
                    frame_number += 1
                    continue

                sampled_frames += 1

                inference_start = time.perf_counter()
                faces = app.get(frame)
                inference_time += time.perf_counter() - inference_start

                detected_faces += len(faces)
                frame_number += 1

            benchmark_time = time.perf_counter() - benchmark_start

            actual_sampling_fps = (
                sampled_frames / duration if duration > 0 else 0
            )
            inference_fps = (
                sampled_frames / inference_time
                if inference_time > 0 else 0
            )
            avg_faces = (
                detected_faces / sampled_frames
                if sampled_frames else 0
            )

            result = {
                "target_sampling_fps": target_fps,
                "frame_interval": interval,
                "sampled_frames": sampled_frames,
                "detected_faces_total": detected_faces,
                "average_faces_per_sample": avg_faces,
                "decode_sec": decode_time,
                "inference_sec": inference_time,
                "benchmark_sec": benchmark_time,
                "actual_sampling_fps": actual_sampling_fps,
                "gpu_inference_fps": inference_fps,
            }

            video_result["rates"].append(result)

            print(
                "Rate %2d FPS | samples=%4d | faces=%5d | "
                "inference=%8.3fs | effective inference FPS=%7.2f"
                % (
                    target_fps,
                    sampled_frames,
                    detected_faces,
                    inference_time,
                    inference_fps,
                )
            )

        cap.release()
        results["videos"].append(video_result)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results["model_load_sec"] = model_load_time
    results["total_benchmark_sec"] = time.perf_counter() - overall_start

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    banner("PHASE 5.2 COMPLETE")
    print("Results:", RESULTS_PATH)
    print(
        "Total benchmark time: %.3f sec"
        % results["total_benchmark_sec"]
    )


if __name__ == "__main__":
    main()
