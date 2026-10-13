#!/usr/bin/env python3

"""
FACETRACK — PHASE 3
ALL 237 CSBS IMAGE GPU RECOGNITION VALIDATION

Purpose
-------
Validate the complete CSBS student enrollment dataset against the
237-vector student FAISS index.

Phase 3 performs:

1. Validate required project files.
2. Validate CUDAExecutionProvider availability.
3. Load buffalo_s through InsightFace.
4. Load the existing 237-vector FAISS index.
5. Load the corresponding metadata.
6. Scan every CSBS student image.
7. Detect faces using the GPU-backed InsightFace pipeline.
8. Generate 512-D ArcFace embeddings.
9. Search the existing FAISS student index.
10. Compare predicted identity with the source folder identity.
11. Record every image-level result.
12. Calculate overall recognition accuracy.
13. Calculate per-student accuracy.
14. Report mismatches and low-confidence results.
15. Report inference and FAISS timing.
16. Write a validation report.

IMPORTANT
---------
This is a READ-ONLY validation program.

It does NOT:
- modify attendance.db
- modify faculty_index.faiss
- modify students.faiss
- modify students.json
- enroll new students
- delete images
- alter source images

Output:
--------
student_index/
    phase3_all_237_results.json
    phase3_all_237_summary.txt
"""


import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import faiss
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path.home()
    / "facetrack"
    / "facetrack_project_complete"
)

DATASET_ROOT = (
    Path.home()
    / "facetrack"
    / "CSBS_students"
)

MODEL_ROOT = PROJECT_ROOT / "models"

INDEX_PATH = (
    PROJECT_ROOT
    / "student_index"
    / "students.faiss"
)

META_PATH = (
    PROJECT_ROOT
    / "student_index"
    / "students.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "student_index"
)

RESULTS_PATH = (
    OUTPUT_ROOT
    / "phase3_all_237_results.json"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "phase3_all_237_summary.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
}

DET_SIZE = (
    640,
    640,
)

TOP_K = 5

# This is NOT used as a hard recognition threshold.
# It is only used to flag potentially weak matches.
LOW_CONFIDENCE_THRESHOLD = 0.40


# ============================================================
# DISPLAY HELPERS
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

    banner(
        "FACETRACK — PHASE 3 ALL 237 IMAGE GPU RECOGNITION VALIDATION"
    )

    print("Project :", PROJECT_ROOT)
    print("Dataset :", DATASET_ROOT)
    print("Model   :", MODEL_ROOT)
    print("FAISS   :", INDEX_PATH)
    print("Meta    :", META_PATH)
    print("Results :", RESULTS_PATH)
    print("Summary :", SUMMARY_PATH)

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # REQUIRED PATH VALIDATION
    # ========================================================

    banner("REQUIRED FILE VALIDATION")

    required_paths = [
        DATASET_ROOT,
        MODEL_ROOT,
        INDEX_PATH,
        META_PATH,
    ]

    for path in required_paths:

        if path.exists():

            print(
                "[OK] ",
                path,
            )

        else:

            print()
            print("[FAIL]", path)
            print()
            raise SystemExit(
                "Required path does not exist."
            )

    # ========================================================
    # ONNX RUNTIME PROVIDERS
    # ========================================================

    banner("ONNX RUNTIME PROVIDERS")

    print(
        "ONNX Runtime:",
        ort.__version__,
    )

    providers = ort.get_available_providers()

    for provider in providers:

        print(
            "  ",
            provider,
        )

    if (
        "CUDAExecutionProvider"
        not in providers
    ):

        print()
        print(
            "FAIL: CUDAExecutionProvider is unavailable."
        )
        print(
            "GPU recognition cannot be validated."
        )

        raise SystemExit(1)

    print()
    print(
        "CUDAExecutionProvider: AVAILABLE"
    )

    # ========================================================
    # LOAD FAISS INDEX
    # ========================================================

    banner("LOADING STUDENT FAISS INDEX")

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    print(
        "FAISS vectors  :",
        index.ntotal,
    )

    print(
        "FAISS dimension:",
        index.d,
    )

    print(
        "FAISS metric   :",
        index.metric_type,
    )

    if index.d != 512:

        raise SystemExit(
            "ERROR: FAISS dimension is not 512."
        )

    if index.ntotal == 0:

        raise SystemExit(
            "ERROR: FAISS index is empty."
        )

    # ========================================================
    # LOAD METADATA
    # ========================================================

    banner("LOADING STUDENT METADATA")

    with open(
        META_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)

    print(
        "Metadata entries:",
        len(metadata),
    )

    if len(metadata) != index.ntotal:

        raise SystemExit(
            "ERROR: FAISS vector count and metadata count differ."
        )

    print(
        "FAISS / metadata count: MATCH"
    )

    # ========================================================
    # BUILD VECTOR -> STUDENT LOOKUP
    # ========================================================

    vector_to_student = {}

    for entry in metadata:

        vector_id = int(
            entry["vector_id"]
        )

        student_id = str(
            entry["student_id"]
        )

        vector_to_student[
            vector_id
        ] = student_id

    # ========================================================
    # LOAD INSIGHTFACE
    # ========================================================

    banner("LOADING INSIGHTFACE ON GPU")

    print(
        "Model pack:",
        "buffalo_s",
    )

    model_start = time.perf_counter()

    app = FaceAnalysis(
        name="buffalo_s",
        root=str(MODEL_ROOT),
        allowed_modules=[
            "detection",
            "recognition",
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

    print()
    print(
        "InsightFace model loaded."
    )

    print(
        "Context ID:",
        0,
    )

    print(
        "Detection size:",
        DET_SIZE,
    )

    print(
        "Model load time: %.3f sec"
        % model_load_time
    )

    # ========================================================
    # DISCOVER STUDENT FOLDERS
    # ========================================================

    banner("DISCOVERING CSBS DATASET")

    student_dirs = sorted(
        d
        for d in DATASET_ROOT.iterdir()
        if d.is_dir()
        and d.name.upper() != "MIXED"
    )

    print(
        "Student folders:",
        len(student_dirs),
    )

    if not student_dirs:

        raise SystemExit(
            "ERROR: No student folders found."
        )

    # ========================================================
    # DISCOVER ALL IMAGES
    # ========================================================

    all_images = []

    per_student_expected = {}

    for student_dir in student_dirs:

        images = sorted(
            p
            for p in student_dir.iterdir()
            if (
                p.is_file()
                and p.suffix in IMAGE_EXTENSIONS
            )
        )

        per_student_expected[
            student_dir.name
        ] = len(images)

        for image_path in images:

            all_images.append(
                (
                    student_dir.name,
                    image_path,
                )
            )

    banner("DATASET INVENTORY")

    print(
        "Student folders:",
        len(student_dirs),
    )

    print(
        "Total images   :",
        len(all_images),
    )

    expected_total = sum(
        per_student_expected.values()
    )

    print(
        "Expected total  :",
        expected_total,
    )

    if len(all_images) != 237:

        print()
        print(
            "WARNING:"
        )

        print(
            "The current dataset contains",
            len(all_images),
            "images, not 237."
        )

        print(
            "Phase 3 will nevertheless process ALL"
            " discovered images."
        )

    else:

        print(
            "237-image dataset: CONFIRMED"
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    total_images = 0

    readable_images = 0

    read_failures = 0

    faces_found = 0

    no_face_images = 0

    multiple_face_images = 0

    correct_matches = 0

    wrong_matches = 0

    low_confidence_matches = 0

    invalid_embedding_images = 0

    detection_times = []

    faiss_times = []

    results = []

    per_student_stats = defaultdict(
        lambda: {
            "images": 0,
            "readable": 0,
            "faces_found": 0,
            "correct": 0,
            "wrong": 0,
            "no_face": 0,
            "read_fail": 0,
            "low_confidence": 0,
        }
    )

    # ========================================================
    # PROCESS EVERY IMAGE
    # ========================================================

    banner(
        "PROCESSING ALL CSBS IMAGES"
    )

    processing_start = time.perf_counter()

    for image_number, (
        expected_student,
        image_path,
    ) in enumerate(
        all_images,
        start=1,
    ):

        total_images += 1

        student_stats = (
            per_student_stats[
                expected_student
            ]
        )

        student_stats["images"] += 1

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            read_failures += 1

            student_stats[
                "read_fail"
            ] += 1

            results.append(
                {
                    "image_number": image_number,
                    "source_image": str(
                        image_path
                    ),
                    "expected_student": (
                        expected_student
                    ),
                    "status": "IMAGE_READ_FAILED",
                }
            )

            print(
                "[%03d/%03d] %-25s -> "
                "IMAGE READ FAILED"
                % (
                    image_number,
                    len(all_images),
                    expected_student[:25],
                )
            )

            continue

        readable_images += 1

        student_stats[
            "readable"
        ] += 1

        # ----------------------------------------------------
        # INSIGHTFACE INFERENCE
        # ----------------------------------------------------

        inference_start = (
            time.perf_counter()
        )

        faces = app.get(image)

        inference_time = (
            time.perf_counter()
            - inference_start
        )

        detection_times.append(
            inference_time
        )

        # ----------------------------------------------------
        # NO FACE
        # ----------------------------------------------------

        if not faces:

            no_face_images += 1

            student_stats[
                "no_face"
            ] += 1

            results.append(
                {
                    "image_number": image_number,
                    "source_image": str(
                        image_path
                    ),
                    "expected_student": (
                        expected_student
                    ),
                    "status": "NO_FACE",
                    "inference_time_sec": (
                        inference_time
                    ),
                }
            )

            print(
                "[%03d/%03d] %-25s -> "
                "NO FACE"
                % (
                    image_number,
                    len(all_images),
                    expected_student[:25],
                )
            )

            continue

        # ----------------------------------------------------
        # FACE FOUND
        # ----------------------------------------------------

        faces_found += 1

        student_stats[
            "faces_found"
        ] += 1

        if len(faces) > 1:

            multiple_face_images += 1

        # ----------------------------------------------------
        # SELECT LARGEST FACE
        # ----------------------------------------------------

        face = max(
            faces,
            key=lambda f: (
                (
                    f.bbox[2]
                    - f.bbox[0]
                )
                *
                (
                    f.bbox[3]
                    - f.bbox[1]
                )
            ),
        )

        # ----------------------------------------------------
        # EXTRACT 512-D EMBEDDING
        # ----------------------------------------------------

        embedding = (
            face.normed_embedding
        )

        if embedding is None:

            invalid_embedding_images += 1

            results.append(
                {
                    "image_number": image_number,
                    "source_image": str(
                        image_path
                    ),
                    "expected_student": (
                        expected_student
                    ),
                    "status": (
                        "INVALID_EMBEDDING"
                    ),
                    "inference_time_sec": (
                        inference_time
                    ),
                }
            )

            print(
                "[%03d/%03d] %-25s -> "
                "INVALID EMBEDDING"
                % (
                    image_number,
                    len(all_images),
                    expected_student[:25],
                )
            )

            continue

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if embedding.size != 512:

            invalid_embedding_images += 1

            results.append(
                {
                    "image_number": image_number,
                    "source_image": str(
                        image_path
                    ),
                    "expected_student": (
                        expected_student
                    ),
                    "status": (
                        "INVALID_EMBEDDING_DIMENSION"
                    ),
                    "embedding_dimension": (
                        int(embedding.size)
                    ),
                    "inference_time_sec": (
                        inference_time
                    ),
                }
            )

            print(
                "[%03d/%03d] %-25s -> "
                "INVALID EMBEDDING DIM=%d"
                % (
                    image_number,
                    len(all_images),
                    expected_student[:25],
                    embedding.size,
                )
            )

            continue

        embedding = embedding.reshape(
            1,
            512,
        )

        # ----------------------------------------------------
        # FAISS SEARCH
        # ----------------------------------------------------

        search_start = (
            time.perf_counter()
        )

        similarities, vector_ids = (
            index.search(
                embedding,
                TOP_K,
            )
        )

        search_time = (
            time.perf_counter()
            - search_start
        )

        faiss_times.append(
            search_time
        )

        best_score = float(
            similarities[0][0]
        )

        best_vector_id = int(
            vector_ids[0][0]
        )

        predicted_student = (
            vector_to_student[
                best_vector_id
            ]
        )

        # ----------------------------------------------------
        # TOP-K RESULTS
        # ----------------------------------------------------

        top_k_matches = []

        for rank in range(
            min(
                TOP_K,
                len(vector_ids[0]),
            )
        ):

            vector_id = int(
                vector_ids[0][rank]
            )

            score = float(
                similarities[0][rank]
            )

            student_id = (
                vector_to_student.get(
                    vector_id,
                    "UNKNOWN",
                )
            )

            top_k_matches.append(
                {
                    "rank": rank + 1,
                    "vector_id": vector_id,
                    "student_id": student_id,
                    "similarity": score,
                }
            )

        # ----------------------------------------------------
        # MATCH EVALUATION
        # ----------------------------------------------------

        is_correct = (
            predicted_student
            == expected_student
        )

        is_low_confidence = (
            best_score
            < LOW_CONFIDENCE_THRESHOLD
        )

        if is_correct:

            correct_matches += 1

            student_stats[
                "correct"
            ] += 1

            status = "CORRECT"

        else:

            wrong_matches += 1

            student_stats[
                "wrong"
            ] += 1

            status = "WRONG"

        if is_low_confidence:

            low_confidence_matches += 1

            student_stats[
                "low_confidence"
            ] += 1

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        result = {
            "image_number": image_number,
            "source_image": str(
                image_path
            ),
            "source_filename": (
                image_path.name
            ),
            "expected_student": (
                expected_student
            ),
            "predicted_student": (
                predicted_student
            ),
            "correct": is_correct,
            "status": status,
            "similarity": best_score,
            "low_confidence": (
                is_low_confidence
            ),
            "vector_id": (
                best_vector_id
            ),
            "embedding_dimension": 512,
            "inference_time_sec": (
                inference_time
            ),
            "faiss_search_time_sec": (
                search_time
            ),
            "det_score": float(
                face.det_score
            ),
            "face_bbox": [
                float(x)
                for x in face.bbox
            ],
            "top_k": top_k_matches,
        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # LIVE PROGRESS
        # ----------------------------------------------------

        print(
            "[%03d/%03d] %-25s -> "
            "%-25s score=%.4f %s"
            % (
                image_number,
                len(all_images),
                expected_student[:25],
                predicted_student[:25],
                best_score,
                status,
            )
        )

    processing_time = (
        time.perf_counter()
        - processing_start
    )

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    recognition_attempts = (
        correct_matches
        + wrong_matches
    )

    if recognition_attempts > 0:

        recognition_accuracy = (
            correct_matches
            / recognition_attempts
        ) * 100.0

    else:

        recognition_accuracy = 0.0

    if readable_images > 0:

        face_detection_rate = (
            faces_found
            / readable_images
        ) * 100.0

    else:

        face_detection_rate = 0.0

    if total_images > 0:

        overall_success_rate = (
            correct_matches
            / total_images
        ) * 100.0

    else:

        overall_success_rate = 0.0

    # ========================================================
    # TIMING METRICS
    # ========================================================

    if detection_times:

        avg_inference = (
            sum(detection_times)
            / len(detection_times)
        )

        min_inference = min(
            detection_times
        )

        max_inference = max(
            detection_times
        )

        inference_fps = (
            1.0 / avg_inference
            if avg_inference > 0
            else 0.0
        )

    else:

        avg_inference = 0.0
        min_inference = 0.0
        max_inference = 0.0
        inference_fps = 0.0

    if faiss_times:

        avg_faiss = (
            sum(faiss_times)
            / len(faiss_times)
        )

        min_faiss = min(
            faiss_times
        )

        max_faiss = max(
            faiss_times
        )

    else:

        avg_faiss = 0.0
        min_faiss = 0.0
        max_faiss = 0.0

    overall_time = (
        time.perf_counter()
        - overall_start
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    banner(
        "PHASE 3 DATASET SUMMARY"
    )

    print(
        "Student folders       :",
        len(student_dirs),
    )

    print(
        "Images discovered     :",
        len(all_images),
    )

    print(
        "Images scanned        :",
        total_images,
    )

    print(
        "Readable images       :",
        readable_images,
    )

    print(
        "Read failures         :",
        read_failures,
    )

    print(
        "Faces detected        :",
        faces_found,
    )

    print(
        "No-face images        :",
        no_face_images,
    )

    print(
        "Multiple-face images  :",
        multiple_face_images,
    )

    print(
        "Invalid embeddings    :",
        invalid_embedding_images,
    )

    print(
        "Correct matches       :",
        correct_matches,
    )

    print(
        "Wrong matches         :",
        wrong_matches,
    )

    print(
        "Low-confidence matches:",
        low_confidence_matches,
    )

    print()
    print(
        "Face detection rate   : %.2f%%"
        % face_detection_rate
    )

    print(
        "Recognition accuracy  : %.2f%%"
        % recognition_accuracy
    )

    print(
        "Overall success rate  : %.2f%%"
        % overall_success_rate
    )

    # ========================================================
    # GPU / PIPELINE TIMING
    # ========================================================

    banner(
        "GPU / PIPELINE TIMING"
    )

    print(
        "Model load time       : %.4f sec"
        % model_load_time
    )

    print(
        "Processing time       : %.4f sec"
        % processing_time
    )

    print(
        "Total elapsed time    : %.4f sec"
        % overall_time
    )

    if detection_times:

        print()
        print(
            "Average InsightFace inference: "
            "%.4f sec"
            % avg_inference
        )

        print(
            "Approx inference FPS         : "
            "%.2f"
            % inference_fps
        )

        print(
            "Fastest inference            : "
            "%.4f sec"
            % min_inference
        )

        print(
            "Slowest inference            : "
            "%.4f sec"
            % max_inference
        )

    if faiss_times:

        print()
        print(
            "Average FAISS search         : "
            "%.6f sec"
            % avg_faiss
        )

        print(
            "Fastest FAISS search         : "
            "%.6f sec"
            % min_faiss
        )

        print(
            "Slowest FAISS search         : "
            "%.6f sec"
            % max_faiss
        )

    # ========================================================
    # MISMATCH REPORT
    # ========================================================

    banner(
        "MISMATCH ANALYSIS"
    )

    mismatches = [
        result
        for result in results
        if result.get("status")
        == "WRONG"
    ]

    if not mismatches:

        print(
            "No identity mismatches."
        )

    else:

        print(
            "Identity mismatches:",
            len(mismatches),
        )

        for result in mismatches:

            print()
            print(
                "Image:",
                result["source_filename"],
            )

            print(
                "Expected:",
                result["expected_student"],
            )

            print(
                "Predicted:",
                result["predicted_student"],
            )

            print(
                "Similarity:",
                "%.4f"
                % result["similarity"],
            )

            print(
                "Top-5:"
            )

            for match in result[
                "top_k"
            ]:

                print(
                    "  #%d %-25s %.4f"
                    % (
                        match["rank"],
                        match["student_id"][:25],
                        match["similarity"],
                    )
                )

    # ========================================================
    # PER-STUDENT RESULTS
    # ========================================================

    banner(
        "PER-STUDENT RECOGNITION"
    )

    per_student_output = {}

    for student_dir in student_dirs:

        student_name = (
            student_dir.name
        )

        stats = per_student_stats[
            student_name
        ]

        attempts = (
            stats["correct"]
            + stats["wrong"]
        )

        if attempts > 0:

            accuracy = (
                stats["correct"]
                / attempts
            ) * 100.0

        else:

            accuracy = 0.0

        per_student_output[
            student_name
        ] = {
            **stats,
            "accuracy_percent": accuracy,
        }

        print(
            "%-30s images=%3d "
            "faces=%3d "
            "correct=%3d "
            "wrong=%3d "
            "accuracy=%6.2f%%"
            % (
                student_name[:30],
                stats["images"],
                stats["faces_found"],
                stats["correct"],
                stats["wrong"],
                accuracy,
            )
        )

    # ========================================================
    # SAVE JSON RESULTS
    # ========================================================

    banner(
        "WRITING PHASE 3 RESULTS"
    )

    report = {
        "phase": 3,
        "description": (
            "All CSBS image GPU recognition validation"
        ),
        "project_root": str(
            PROJECT_ROOT
        ),
        "dataset_root": str(
            DATASET_ROOT
        ),
        "model_root": str(
            MODEL_ROOT
        ),
        "faiss_index": str(
            INDEX_PATH
        ),
        "metadata": str(
            META_PATH
        ),
        "model": "buffalo_s",
        "context_id": 0,
        "detection_size": list(
            DET_SIZE
        ),
        "faiss_vectors": int(
            index.ntotal
        ),
        "faiss_dimension": int(
            index.d
        ),
        "onnxruntime_version": (
            ort.__version__
        ),
        "onnxruntime_providers": providers,
        "dataset": {
            "student_folders": len(
                student_dirs
            ),
            "images_discovered": len(
                all_images
            ),
            "images_scanned": total_images,
            "readable_images": (
                readable_images
            ),
            "read_failures": (
                read_failures
            ),
            "faces_detected": (
                faces_found
            ),
            "no_face_images": (
                no_face_images
            ),
            "multiple_face_images": (
                multiple_face_images
            ),
            "invalid_embeddings": (
                invalid_embedding_images
            ),
        },
        "recognition": {
            "correct_matches": (
                correct_matches
            ),
            "wrong_matches": (
                wrong_matches
            ),
            "recognition_attempts": (
                recognition_attempts
            ),
            "recognition_accuracy_percent": (
                recognition_accuracy
            ),
            "face_detection_rate_percent": (
                face_detection_rate
            ),
            "overall_success_rate_percent": (
                overall_success_rate
            ),
            "low_confidence_matches": (
                low_confidence_matches
            ),
            "low_confidence_threshold": (
                LOW_CONFIDENCE_THRESHOLD
            ),
        },
        "timing": {
            "model_load_sec": (
                model_load_time
            ),
            "processing_sec": (
                processing_time
            ),
            "total_elapsed_sec": (
                overall_time
            ),
            "average_inference_sec": (
                avg_inference
            ),
            "inference_fps": (
                inference_fps
            ),
            "fastest_inference_sec": (
                min_inference
            ),
            "slowest_inference_sec": (
                max_inference
            ),
            "average_faiss_search_sec": (
                avg_faiss
            ),
            "fastest_faiss_search_sec": (
                min_faiss
            ),
            "slowest_faiss_search_sec": (
                max_faiss
            ),
        },
        "per_student": (
            per_student_output
        ),
        "results": results,
        "integrity": {
            "student_index_modified": False,
            "student_metadata_modified": False,
            "attendance_db_modified": False,
            "faculty_index_modified": False,
            "source_images_modified": False,
        },
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print(
        "JSON results:",
        RESULTS_PATH,
    )

    # ========================================================
    # SAVE HUMAN-READABLE SUMMARY
    # ========================================================

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "FACETRACK — PHASE 3\n"
        )

        f.write(
            "ALL 237 CSBS IMAGE GPU "
            "RECOGNITION VALIDATION\n"
        )

        f.write(
            "=" * 78
            + "\n\n"
        )

        f.write(
            "Student folders       : %d\n"
            % len(student_dirs)
        )

        f.write(
            "Images discovered     : %d\n"
            % len(all_images)
        )

        f.write(
            "Images scanned        : %d\n"
            % total_images
        )

        f.write(
            "Readable images       : %d\n"
            % readable_images
        )

        f.write(
            "Read failures         : %d\n"
            % read_failures
        )

        f.write(
            "Faces detected        : %d\n"
            % faces_found
        )

        f.write(
            "No-face images        : %d\n"
            % no_face_images
        )

        f.write(
            "Multiple-face images  : %d\n"
            % multiple_face_images
        )

        f.write(
            "Invalid embeddings    : %d\n"
            % invalid_embedding_images
        )

        f.write(
            "Correct matches       : %d\n"
            % correct_matches
        )

        f.write(
            "Wrong matches         : %d\n"
            % wrong_matches
        )

        f.write(
            "Low-confidence matches: %d\n"
            % low_confidence_matches
        )

        f.write(
            "\n"
        )

        f.write(
            "Face detection rate   : %.2f%%\n"
            % face_detection_rate
        )

        f.write(
            "Recognition accuracy  : %.2f%%\n"
            % recognition_accuracy
        )

        f.write(
            "Overall success rate  : %.2f%%\n"
            % overall_success_rate
        )

        f.write(
            "\n"
        )

        f.write(
            "Average inference     : %.4f sec\n"
            % avg_inference
        )

        f.write(
            "Approx inference FPS  : %.2f\n"
            % inference_fps
        )

        f.write(
            "Average FAISS search  : %.6f sec\n"
            % avg_faiss
        )

        f.write(
            "Processing time       : %.4f sec\n"
            % processing_time
        )

        f.write(
            "\n"
        )

        f.write(
            "MISMATCHES\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        if not mismatches:

            f.write(
                "None\n"
            )

        else:

            for result in mismatches:

                f.write(
                    "\n"
                )

                f.write(
                    "Image: %s\n"
                    % result[
                        "source_filename"
                    ]
                )

                f.write(
                    "Expected: %s\n"
                    % result[
                        "expected_student"
                    ]
                )

                f.write(
                    "Predicted: %s\n"
                    % result[
                        "predicted_student"
                    ]
                )

                f.write(
                    "Similarity: %.4f\n"
                    % result[
                        "similarity"
                    ]
                )

        f.write(
            "\n"
        )

        f.write(
            "INTEGRITY\n"
        )

        f.write(
            "-" * 78
            + "\n"
        )

        f.write(
            "Student index modified : NO\n"
        )

        f.write(
            "Student metadata modified : NO\n"
        )

        f.write(
            "Attendance DB modified : NO\n"
        )

        f.write(
            "Faculty index modified : NO\n"
        )

        f.write(
            "Source images modified : NO\n"
        )

    print(
        "Text summary:",
        SUMMARY_PATH,
    )

    # ========================================================
    # FINAL VERDICT
    # ========================================================

    banner(
        "PHASE 3 VERDICT"
    )

    if (
        total_images == 237
        and readable_images == 237
        and faces_found == 237
        and correct_matches == 237
    ):

        print(
            "PASS"
        )

        print(
            "All 237 images were readable,"
            " faces were detected,"
            " and every image matched"
            " its expected student identity."
        )

    elif (
        recognition_attempts > 0
        and correct_matches
        == recognition_attempts
        and no_face_images == 0
    ):

        print(
            "PASS WITH DATASET DIFFERENCE"
        )

        print(
            "All attempted recognitions were"
            " correct, but the discovered"
            " dataset differs from exactly"
            " 237 images."
        )

    elif (
        recognition_attempts > 0
        and correct_matches
        == recognition_attempts
    ):

        print(
            "PARTIAL PASS"
        )

        print(
            "Every image that reached recognition"
            " matched correctly, but some images"
            " could not be recognized."
        )

    elif (
        correct_matches > 0
    ):

        print(
            "PARTIAL PASS"
        )

        print(
            "Recognition is functioning, but"
            " some images produced failures"
            " or identity mismatches."
        )

    else:

        print(
            "FAIL"
        )

        print(
            "No correct identity matches were"
            " established."
        )

    print()
    print(
        "Student index modified       : NO"
    )

    print(
        "Student metadata modified    : NO"
    )

    print(
        "Attendance DB modified       : NO"
    )

    print(
        "Faculty index modified       : NO"
    )

    print(
        "Source images modified       : NO"
    )

    print()
    print(
        "=" * 78
    )

    print(
        "PHASE 3 COMPLETE"
    )

    print(
        "=" * 78
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
