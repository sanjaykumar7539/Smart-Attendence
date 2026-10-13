import time
import json
from pathlib import Path

import cv2
import faiss
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis


# ============================================================
# FACETRACK — PHASE 5.4
# RAW 2-FPS CLASSROOM RECOGNITION BASELINE
# ============================================================

PROJECT_ROOT = Path.home() / "facetrack" / "facetrack_project_complete"

VIDEO_ROOT = Path.home() / "facetrack" / "phase5_validation"

INDEX_PATH = PROJECT_ROOT / "student_index" / "students.faiss"

METADATA_PATH = PROJECT_ROOT / "student_index" / "students.json"

RESULTS_DIR = PROJECT_ROOT / "student_index"

RESULTS_PATH = RESULTS_DIR / "phase5_4_raw_recognition_2fps.json"

MODEL_ROOT = PROJECT_ROOT / "models"

MODEL_NAME = "buffalo_s"

DET_SIZE = (640, 640)

SOURCE_FPS_EXPECTED = 20.0

TARGET_FPS = 2.0

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
}


# ============================================================
# HELPERS
# ============================================================

def banner(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = time.perf_counter()

    banner("FACETRACK — PHASE 5.4 RAW 2-FPS RECOGNITION")

    print("Video root :", VIDEO_ROOT)
    print("FAISS      :", INDEX_PATH)
    print("Metadata   :", METADATA_PATH)
    print("Model root :", MODEL_ROOT)
    print("Model      :", MODEL_NAME)
    print("Det size   :", DET_SIZE)
    print("Target FPS :", TARGET_FPS)

    # --------------------------------------------------------
    # ONNX RUNTIME VALIDATION
    # --------------------------------------------------------

    banner("ONNX RUNTIME VALIDATION")

    providers = ort.get_available_providers()

    for provider in providers:
        print("  ", provider)

    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable."
        )

    print()
    print("CUDAExecutionProvider: AVAILABLE")

    # --------------------------------------------------------
    # LOAD FAISS
    # --------------------------------------------------------

    banner("LOADING EXISTING FAISS INDEX")

    index = faiss.read_index(str(INDEX_PATH))

    print("FAISS vectors :", index.ntotal)
    print("FAISS dimension:", index.d)
    print("FAISS metric  :", index.metric_type)

    if index.ntotal != 237:
        raise RuntimeError(
            "Expected 237 FAISS vectors."
        )

    if index.d != 512:
        raise RuntimeError(
            "Expected 512-D FAISS embeddings."
        )

    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise RuntimeError(
            "Expected INNER_PRODUCT FAISS metric."
        )

    # --------------------------------------------------------
    # LOAD METADATA
    # --------------------------------------------------------

    banner("LOADING STUDENT METADATA")

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    if not isinstance(metadata, list):
        raise RuntimeError(
            "students.json must contain a list."
        )

    if len(metadata) != index.ntotal:
        raise RuntimeError(
            "FAISS / metadata count mismatch."
        )

    metadata_by_vector_id = {}

    for entry in metadata:

        vector_id = int(entry["vector_id"])

        metadata_by_vector_id[vector_id] = entry

    print(
        "Metadata entries:",
        len(metadata_by_vector_id),
    )

    # --------------------------------------------------------
    # LOAD INSIGHTFACE
    # --------------------------------------------------------

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

    app.prepare(
        ctx_id=0,
        det_size=DET_SIZE,
    )

    model_load_time = (
        time.perf_counter()
        - model_start
    )

    print(
        "Model load time: %.4f sec"
        % model_load_time
    )

    # --------------------------------------------------------
    # VIDEO DISCOVERY
    # --------------------------------------------------------

    videos = sorted(
        p
        for p in VIDEO_ROOT.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in VIDEO_EXTENSIONS
    )

    if len(videos) != 6:

        print(
            "WARNING: Expected 6 validation videos."
        )

    print()
    print(
        "Validation videos:",
        len(videos),
    )

    for video in videos:
        print("  ", video.name)

    # --------------------------------------------------------
    # RESULT STRUCTURE
    # --------------------------------------------------------

    results = {

        "phase": "5.4",

        "description":
            "Raw classroom face recognition baseline at exactly 2 FPS",

        "model": MODEL_NAME,

        "det_size":
            list(DET_SIZE),

        "providers":
            providers,

        "faiss_index":
            str(INDEX_PATH),

        "faiss_vectors":
            int(index.ntotal),

        "faiss_dimension":
            int(index.d),

        "faiss_metric":
            "INNER_PRODUCT",

        "target_sampling_fps":
            TARGET_FPS,

        "source_fps_expected":
            SOURCE_FPS_EXPECTED,

        "temporal_voting":
            False,

        "recognition_threshold":
            None,

        "model_load_sec":
            model_load_time,

        "videos": [],
    }

    # --------------------------------------------------------
    # PROCESS EACH VIDEO
    # --------------------------------------------------------

    for video_path in videos:

        banner(
            "VIDEO: %s"
            % video_path.name
        )

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():

            print(
                "ERROR: Could not open video."
            )

            continue

        source_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        duration = (
            total_frames / source_fps
            if source_fps > 0
            else 0
        )

        frame_interval = max(
            1,
            int(
                round(
                    source_fps
                    / TARGET_FPS
                )
            ),
        )

        print(
            "Source FPS     : %.2f"
            % source_fps
        )

        print(
            "Total frames   : %d"
            % total_frames
        )

        print(
            "Duration       : %.2f sec"
            % duration
        )

        print(
            "Frame interval : %d"
            % frame_interval
        )

        video_result = {

            "video":
                video_path.name,

            "source_fps":
                source_fps,

            "total_frames":
                total_frames,

            "duration_sec":
                duration,

            "target_sampling_fps":
                TARGET_FPS,

            "frame_interval":
                frame_interval,

            "frames": [],
        }

        frame_number = 0

        sampled_frames = 0

        detected_faces = 0

        recognized_faces = 0

        video_start = time.perf_counter()

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            # ------------------------------------------------
            # EXACT 2-FPS SAMPLING
            # ------------------------------------------------

            if frame_number % frame_interval != 0:

                frame_number += 1

                continue

            sampled_frames += 1

            timestamp_sec = (
                frame_number / source_fps
                if source_fps > 0
                else 0
            )

            # ------------------------------------------------
            # FACE DETECTION + EMBEDDING
            # ------------------------------------------------

            inference_start = (
                time.perf_counter()
            )

            faces = app.get(frame)

            inference_sec = (
                time.perf_counter()
                - inference_start
            )

            detected_faces += len(faces)

            frame_record = {

                "frame_number":
                    frame_number,

                "timestamp_sec":
                    timestamp_sec,

                "face_count":
                    len(faces),

                "faces": [],
            }

            # ------------------------------------------------
            # FAISS RECOGNITION
            # ------------------------------------------------

            for face_index, face in enumerate(
                faces
            ):

                embedding = getattr(
                    face,
                    "normed_embedding",
                    None,
                )

                face_record = {

                    "face_index":
                        face_index,

                    "bbox":
                        [
                            float(x)
                            for x in face.bbox
                        ],

                    "det_score":
                        float(face.det_score),

                    "embedding_dimension":
                        None,

                    "embedding_norm":
                        None,

                    "faiss_vector_id":
                        None,

                    "student_id":
                        None,

                    "student_name":
                        None,

                    "similarity":
                        None,

                    "recognized":
                        False,
                }

                if embedding is None:

                    face_record[
                        "embedding_error"
                    ] = "missing_embedding"

                    frame_record[
                        "faces"
                    ].append(
                        face_record
                    )

                    continue

                embedding = np.asarray(
                    embedding,
                    dtype="float32",
                ).reshape(
                    1,
                    -1,
                )

                face_record[
                    "embedding_dimension"
                ] = int(
                    embedding.shape[1]
                )

                face_record[
                    "embedding_norm"
                ] = float(
                    np.linalg.norm(
                        embedding
                    )
                )

                if embedding.shape[1] != 512:

                    face_record[
                        "embedding_error"
                    ] = (
                        "unexpected_dimension"
                    )

                    frame_record[
                        "faces"
                    ].append(
                        face_record
                    )

                    continue

                distances, indices = (
                    index.search(
                        embedding,
                        1,
                    )
                )

                vector_id = int(
                    indices[0][0]
                )

                similarity = float(
                    distances[0][0]
                )

                face_record[
                    "faiss_vector_id"
                ] = vector_id

                face_record[
                    "similarity"
                ] = similarity

                if vector_id >= 0:

                    entry = (
                        metadata_by_vector_id
                        .get(vector_id)
                    )

                    if entry is not None:

                        face_record[
                            "student_id"
                        ] = entry.get(
                            "student_id"
                        )

                        face_record[
                            "student_name"
                        ] = entry.get(
                            "student_name"
                        )

                        face_record[
                            "recognized"
                        ] = True

                        recognized_faces += 1

                frame_record[
                    "faces"
                ].append(
                    face_record
                )

            video_result[
                "frames"
            ].append(
                frame_record
            )

            print(
                "frame=%4d  "
                "time=%6.2fs  "
                "faces=%2d  "
                "inference=%7.4fs"
                % (
                    frame_number,
                    timestamp_sec,
                    len(faces),
                    inference_sec,
                )
            )

            frame_number += 1

        cap.release()

        video_elapsed = (
            time.perf_counter()
            - video_start
        )

        video_result[
            "sampled_frames"
        ] = sampled_frames

        video_result[
            "detected_faces_total"
        ] = detected_faces

        video_result[
            "recognized_faces_total"
        ] = recognized_faces

        video_result[
            "processing_wall_time_sec"
        ] = video_elapsed

        video_result[
            "effective_sampling_fps"
        ] = (
            sampled_frames / duration
            if duration > 0
            else 0
        )

        results[
            "videos"
        ].append(
            video_result
        )

        print()
        print(
            "Sampled frames     : %d"
            % sampled_frames
        )

        print(
            "Detected faces     : %d"
            % detected_faces
        )

        print(
            "Recognized faces   : %d"
            % recognized_faces
        )

        print(
            "Processing time    : %.3f sec"
            % video_elapsed
        )

    # --------------------------------------------------------
    # GLOBAL SUMMARY
    # --------------------------------------------------------

    total_sampled_frames = sum(
        v["sampled_frames"]
        for v in results["videos"]
    )

    total_detected_faces = sum(
        v["detected_faces_total"]
        for v in results["videos"]
    )

    total_recognized_faces = sum(
        v["recognized_faces_total"]
        for v in results["videos"]
    )

    results[
        "summary"
    ] = {

        "videos_processed":
            len(results["videos"]),

        "sampled_frames_total":
            total_sampled_frames,

        "detected_faces_total":
            total_detected_faces,

        "recognized_faces_total":
            total_recognized_faces,

        "recognition_assignment_rate":
            (
                total_recognized_faces
                / total_detected_faces
                if total_detected_faces
                else 0
            ),
    }

    results[
        "total_runtime_sec"
    ] = (
        time.perf_counter()
        - overall_start
    )

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    banner("PHASE 5.4 COMPLETE")

    print(
        "Videos processed      : %d"
        % len(results["videos"])
    )

    print(
        "Sampled frames        : %d"
        % total_sampled_frames
    )

    print(
        "Detected faces        : %d"
        % total_detected_faces
    )

    print(
        "Recognition assignments: %d"
        % total_recognized_faces
    )

    print(
        "Assignment rate       : %.4f"
        % results[
            "summary"
        ][
            "recognition_assignment_rate"
        ]
    )

    print(
        "Results:"
    )

    print(
        RESULTS_PATH
    )


if __name__ == "__main__":
    main()
