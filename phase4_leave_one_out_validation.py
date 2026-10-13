#!/usr/bin/env python3

import json
import time
from pathlib import Path

import cv2
import faiss
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis


# ============================================================
# FACETRACK — PHASE 4
# LEAVE-ONE-OUT / CROSS-IMAGE RECOGNITION VALIDATION
#
# PURPOSE
# -------
# Validate genuine cross-image recognition.
#
# Unlike Phase 3:
#
#     TEST IMAGE
#         ↓
#     FAISS
#         ↓
#     OWN ENROLLMENT VECTOR
#
# Phase 4 explicitly excludes the test image's own vector.
#
# Therefore:
#
#     TEST IMAGE
#         ↓
#     GPU FACE DETECTION
#         ↓
#     512-D ARCFace EMBEDDING
#         ↓
#     FAISS
#         ↓
#     EXCLUDE OWN VECTOR
#         ↓
#     NEAREST OTHER IMAGE
#         ↓
#     PREDICTED STUDENT
#
# INPUT
# -----
# ~/facetrack/CSBS_students/
#
# EXISTING INDEX
# --------------
# student_index/students.faiss
# student_index/students.json
#
# OUTPUT
# ------
# student_index/
#     phase4_leave_one_out_results.json
#     phase4_leave_one_out_summary.txt
#
# SAFETY
# ------
# This program DOES NOT modify:
#
#     attendance.db
#     faculty index
#     students.faiss
#     students.json
#     source images
#
# ============================================================


# ============================================================
# PATHS
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

MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
)

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

RESULTS_PATH = (
    PROJECT_ROOT
    / "student_index"
    / "phase4_leave_one_out_results.json"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "student_index"
    / "phase4_leave_one_out_summary.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

DET_SIZE = (640, 640)

# We request more than one result because the test image's
# own vector must be skipped.
TOP_K = 10

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
}


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
        "FACETRACK — PHASE 4 "
        "LEAVE-ONE-OUT GPU RECOGNITION VALIDATION"
    )

    print("Project :", PROJECT_ROOT)
    print("Dataset :", DATASET_ROOT)
    print("Model   :", MODEL_ROOT)
    print("FAISS   :", INDEX_PATH)
    print("Meta    :", META_PATH)
    print("Results :", RESULTS_PATH)
    print("Summary :", SUMMARY_PATH)

    # ========================================================
    # BASIC PATH VALIDATION
    # ========================================================

    banner("VALIDATING REQUIRED PATHS")

    required_paths = [
        DATASET_ROOT,
        MODEL_ROOT,
        INDEX_PATH,
        META_PATH,
    ]

    for path in required_paths:

        print(
            "%-10s : %s"
            % (
                "EXISTS" if path.exists() else "MISSING",
                path,
            )
        )

        if not path.exists():

            print()
            print("ERROR: Required path does not exist.")
            raise SystemExit(1)

    # ========================================================
    # CUDA VALIDATION
    # ========================================================

    banner("ONNX RUNTIME / CUDA VALIDATION")

    print(
        "ONNX Runtime:",
        ort.__version__,
    )

    providers = ort.get_available_providers()

    print("Available providers:")

    for provider in providers:

        print(
            "  ",
            provider,
        )

    if "CUDAExecutionProvider" not in providers:

        print()
        print(
            "ERROR: CUDAExecutionProvider unavailable."
        )

        raise SystemExit(1)

    print()
    print(
        "CUDAExecutionProvider: AVAILABLE"
    )

    # ========================================================
    # LOAD FAISS
    # ========================================================

    banner("LOADING EXISTING STUDENT FAISS INDEX")

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

    if index.ntotal != 237:

        print()
        print(
            "ERROR: Expected 237 FAISS vectors."
        )

        print(
            "Found:",
            index.ntotal,
        )

        raise SystemExit(1)

    if index.d != 512:

        print()
        print(
            "ERROR: Expected 512-D embeddings."
        )

        raise SystemExit(1)

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

        print()
        print(
            "ERROR: FAISS / metadata mismatch."
        )

        raise SystemExit(1)

    # ========================================================
    # BUILD VECTOR LOOKUP
    #
    # Every enrollment image has one corresponding vector.
    #
    # We map:
    #
    # normalized source image path
    #              ↓
    # vector_id
    #
    # This lets us exclude the test image's own vector.
    # ========================================================

    banner("BUILDING VECTOR / IMAGE LOOKUP")

    image_to_vector = {}

    for item in metadata:

        vector_id = int(
            item["vector_id"]
        )

        source_image = str(
            item["source_image"]
        )

        image_to_vector[
            source_image
        ] = vector_id

        # Also store resolved path where possible.
        try:

            resolved = str(
                Path(source_image).resolve()
            )

            image_to_vector[
                resolved
            ] = vector_id

        except Exception:

            pass

    print(
        "Image → vector mappings:",
        len(image_to_vector),
    )

    # ========================================================
    # LOAD INSIGHTFACE
    # ========================================================

    banner("LOADING INSIGHTFACE ON GPU")

    app = FaceAnalysis(
        name="buffalo_s",
        root=str(MODEL_ROOT),
        allowed_modules=[
            "detection",
            "recognition",
        ],
    )

    model_start = time.perf_counter()

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
        "Context ID      : 0"
    )

    print(
        "Detection size  :",
        DET_SIZE,
    )

    print(
        "Model load time : %.4f sec"
        % model_load_time
    )

    # ========================================================
    # DISCOVER STUDENT DIRECTORIES
    #
    # MIXED is deliberately excluded.
    # ========================================================

    banner("DISCOVERING STUDENT DATASET")

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

    if len(student_dirs) != 57:

        print()
        print(
            "WARNING: Expected 57 student folders."
        )

    # ========================================================
    # BUILD TEST IMAGE LIST
    # ========================================================

    all_images = []

    for student_dir in student_dirs:

        images = sorted(
            p
            for p in student_dir.iterdir()
            if p.is_file()
            and p.suffix in IMAGE_EXTENSIONS
        )

        for image_path in images:

            all_images.append(
                (
                    student_dir.name,
                    image_path,
                )
            )

    print(
        "Student images:",
        len(all_images),
    )

    if len(all_images) != 237:

        print()
        print(
            "ERROR: Expected exactly 237 "
            "student images."
        )

        print(
            "Found:",
            len(all_images),
        )

        raise SystemExit(1)

    # ========================================================
    # STATISTICS
    # ========================================================

    total_images = 0

    readable_images = 0

    read_failures = 0

    faces_found = 0

    no_face_images = 0

    multiple_face_images = 0

    invalid_embeddings = 0

    recognition_attempts = 0

    correct_matches = 0

    wrong_matches = 0

    own_vector_excluded = 0

    exclusion_failures = 0

    detection_times = []

    faiss_times = []

    similarity_scores = []

    results = []

    per_student = {}

    # ========================================================
    # INITIALIZE PER-STUDENT STATISTICS
    # ========================================================

    for student_dir in student_dirs:

        per_student[
            student_dir.name
        ] = {

            "images": 0,

            "readable": 0,

            "faces_found": 0,

            "correct": 0,

            "wrong": 0,

            "no_face": 0,

            "read_fail": 0,

            "invalid_embedding": 0,

            "own_vector_excluded": 0,

            "exclusion_failure": 0,

            "accuracy_percent": 0.0,

        }

    # ========================================================
    # PROCESS EVERY IMAGE
    # ========================================================

    banner(
        "RUNNING 237-IMAGE LEAVE-ONE-OUT VALIDATION"
    )

    for image_number, pair in enumerate(
        all_images,
        start=1,
    ):

        expected_student, image_path = pair

        total_images += 1

        per_student[
            expected_student
        ]["images"] += 1

        # ----------------------------------------------------
        # IMAGE PATH / VECTOR ID
        # ----------------------------------------------------

        image_path_resolved = str(
            image_path.resolve()
        )

        own_vector_id = image_to_vector.get(
            image_path_resolved
        )

        if own_vector_id is None:

            own_vector_id = image_to_vector.get(
                str(image_path)
            )

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            read_failures += 1

            per_student[
                expected_student
            ]["read_fail"] += 1

            print(
                "[%03d/%03d] %-28s -> READ FAILED"
                % (
                    image_number,
                    len(all_images),
                    expected_student[:28],
                )
            )

            results.append({

                "image_number":
                    image_number,

                "expected_student":
                    expected_student,

                "image":
                    str(image_path),

                "status":
                    "READ_FAILED",

                "own_vector_id":
                    own_vector_id,

            })

            continue

        readable_images += 1

        per_student[
            expected_student
        ]["readable"] += 1

        # ----------------------------------------------------
        # GPU FACE DETECTION + RECOGNITION
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

            per_student[
                expected_student
            ]["no_face"] += 1

            print(
                "[%03d/%03d] %-28s -> NO FACE"
                % (
                    image_number,
                    len(all_images),
                    expected_student[:28],
                )
            )

            results.append({

                "image_number":
                    image_number,

                "expected_student":
                    expected_student,

                "image":
                    str(image_path),

                "status":
                    "NO_FACE",

                "inference_sec":
                    inference_time,

                "own_vector_id":
                    own_vector_id,

            })

            continue

        faces_found += 1

        per_student[
            expected_student
        ]["faces_found"] += 1

        # ----------------------------------------------------
        # MULTIPLE FACES
        #
        # We still select the largest face, but record this
        # condition explicitly.
        # ----------------------------------------------------

        if len(faces) > 1:

            multiple_face_images += 1

        # ----------------------------------------------------
        # SELECT LARGEST FACE
        # ----------------------------------------------------

        face = max(
            faces,
            key=lambda f:
            (
                f.bbox[2] - f.bbox[0]
            )
            *
            (
                f.bbox[3] - f.bbox[1]
            ),
        )

        # ----------------------------------------------------
        # EXTRACT NORMALIZED 512-D EMBEDDING
        # ----------------------------------------------------

        embedding = (
            face.normed_embedding
            .astype(np.float32)
        )

        if embedding is None:

            invalid_embeddings += 1

            per_student[
                expected_student
            ]["invalid_embedding"] += 1

            print(
                "[%03d/%03d] %-28s -> "
                "INVALID EMBEDDING"
                % (
                    image_number,
                    len(all_images),
                    expected_student[:28],
                )
            )

            results.append({

                "image_number":
                    image_number,

                "expected_student":
                    expected_student,

                "image":
                    str(image_path),

                "status":
                    "INVALID_EMBEDDING",

                "inference_sec":
                    inference_time,

                "own_vector_id":
                    own_vector_id,

            })

            continue

        embedding = embedding.reshape(
            1,
            -1,
        )

        if embedding.shape[1] != 512:

            invalid_embeddings += 1

            per_student[
                expected_student
            ]["invalid_embedding"] += 1

            print(
                "[%03d/%03d] %-28s -> "
                "INVALID DIMENSION: %s"
                % (
                    image_number,
                    len(all_images),
                    expected_student[:28],
                    embedding.shape,
                )
            )

            results.append({

                "image_number":
                    image_number,

                "expected_student":
                    expected_student,

                "image":
                    str(image_path),

                "status":
                    "INVALID_DIMENSION",

                "embedding_dimension":
                    int(embedding.shape[1]),

                "inference_sec":
                    inference_time,

                "own_vector_id":
                    own_vector_id,

            })

            continue

        # ----------------------------------------------------
        # FAISS SEARCH
        #
        # Search more candidates than required because the
        # test image's own vector must be excluded.
        # ----------------------------------------------------

        recognition_attempts += 1

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

        # ----------------------------------------------------
        # FIND FIRST VECTOR THAT IS NOT THE TEST IMAGE
        # ----------------------------------------------------

        selected_score = None

        selected_vector_id = None

        selected_metadata = None

        own_excluded = False

        for rank in range(
            vector_ids.shape[1]
        ):

            candidate_vector_id = int(
                vector_ids[0][rank]
            )

            candidate_score = float(
                similarities[0][rank]
            )

            if (
                own_vector_id is not None
                and candidate_vector_id
                == own_vector_id
            ):

                own_excluded = True

                continue

            selected_vector_id = (
                candidate_vector_id
            )

            selected_score = (
                candidate_score
            )

            selected_metadata = (
                metadata[
                    selected_vector_id
                ]
            )

            break

        # ----------------------------------------------------
        # OWN VECTOR EXCLUSION STATUS
        # ----------------------------------------------------

        if own_excluded:

            own_vector_excluded += 1

            per_student[
                expected_student
            ]["own_vector_excluded"] += 1

        else:

            exclusion_failures += 1

            per_student[
                expected_student
            ]["exclusion_failure"] += 1

        # ----------------------------------------------------
        # NO OTHER VECTOR AVAILABLE
        # ----------------------------------------------------

        if selected_metadata is None:

            print(
                "[%03d/%03d] %-28s -> "
                "NO ALTERNATE VECTOR"
                % (
                    image_number,
                    len(all_images),
                    expected_student[:28],
                )
            )

            results.append({

                "image_number":
                    image_number,

                "expected_student":
                    expected_student,

                "image":
                    str(image_path),

                "status":
                    "NO_ALTERNATE_VECTOR",

                "own_vector_id":
                    own_vector_id,

                "own_vector_excluded":
                    own_excluded,

                "inference_sec":
                    inference_time,

                "faiss_search_sec":
                    search_time,

            })

            continue

        # ----------------------------------------------------
        # PREDICTED STUDENT
        # ----------------------------------------------------

        predicted_student = (
            selected_metadata[
                "student_id"
            ]
        )

        correct = (
            predicted_student
            == expected_student
        )

        if correct:

            correct_matches += 1

            per_student[
                expected_student
            ]["correct"] += 1

            status = "CORRECT"

        else:

            wrong_matches += 1

            per_student[
                expected_student
            ]["wrong"] += 1

            status = "WRONG"

        similarity_scores.append(
            selected_score
        )

        # ----------------------------------------------------
        # SAVE RESULT RECORD
        # ----------------------------------------------------

        result_record = {

            "image_number":
                image_number,

            "expected_student":
                expected_student,

            "predicted_student":
                predicted_student,

            "image":
                str(image_path),

            "status":
                status,

            "own_vector_id":
                own_vector_id,

            "matched_vector_id":
                selected_vector_id,

            "own_vector_excluded":
                own_excluded,

            "similarity":
                selected_score,

            "detection_score":
                float(face.det_score),

            "face_width":
                float(
                    face.bbox[2]
                    - face.bbox[0]
                ),

            "face_height":
                float(
                    face.bbox[3]
                    - face.bbox[1]
                ),

            "inference_sec":
                inference_time,

            "faiss_search_sec":
                search_time,

        }

        results.append(
            result_record
        )

        # ----------------------------------------------------
        # PROGRESS OUTPUT
        # ----------------------------------------------------

        print(
            "[%03d/%03d] %-28s -> %-28s "
            "score=%.4f  %s"
            % (
                image_number,
                len(all_images),
                expected_student[:28],
                predicted_student[:28],
                selected_score,
                status,
            )
        )

    # ========================================================
    # CALCULATE PER-STUDENT ACCURACY
    # ========================================================

    for student_name, stats in (
        per_student.items()
    ):

        attempts = (
            stats["correct"]
            +
            stats["wrong"]
        )

        if attempts > 0:

            stats["accuracy_percent"] = (
                stats["correct"]
                /
                attempts
            ) * 100.0

    # ========================================================
    # GLOBAL METRICS
    # ========================================================

    face_detection_rate = (
        faces_found
        /
        total_images
        *
        100.0
        if total_images
        else 0.0
    )

    recognition_accuracy = (
        correct_matches
        /
        recognition_attempts
        *
        100.0
        if recognition_attempts
        else 0.0
    )

    overall_success_rate = (
        correct_matches
        /
        total_images
        *
        100.0
        if total_images
        else 0.0
    )

    if similarity_scores:

        average_similarity = (
            sum(similarity_scores)
            /
            len(similarity_scores)
        )

        minimum_similarity = min(
            similarity_scores
        )

        maximum_similarity = max(
            similarity_scores
        )

    else:

        average_similarity = 0.0

        minimum_similarity = 0.0

        maximum_similarity = 0.0

    if detection_times:

        average_inference = (
            sum(detection_times)
            /
            len(detection_times)
        )

        inference_fps = (
            1.0
            /
            average_inference
            if average_inference > 0
            else 0.0
        )

        fastest_inference = min(
            detection_times
        )

        slowest_inference = max(
            detection_times
        )

    else:

        average_inference = 0.0

        inference_fps = 0.0

        fastest_inference = 0.0

        slowest_inference = 0.0

    if faiss_times:

        average_faiss = (
            sum(faiss_times)
            /
            len(faiss_times)
        )

        fastest_faiss = min(
            faiss_times
        )

        slowest_faiss = max(
            faiss_times
        )

    else:

        average_faiss = 0.0

        fastest_faiss = 0.0

        slowest_faiss = 0.0

    total_elapsed = (
        time.perf_counter()
        -
        overall_start
    )

    # ========================================================
    # MISMATCH LIST
    # ========================================================

    mismatches = [

        result

        for result in results

        if result.get("status")
        == "WRONG"

    ]

    # ========================================================
    # BUILD JSON REPORT
    # ========================================================

    report = {

        "phase": 4,

        "description":
            "Leave-one-out / cross-image "
            "GPU recognition validation",

        "project_root":
            str(PROJECT_ROOT),

        "dataset_root":
            str(DATASET_ROOT),

        "model_root":
            str(MODEL_ROOT),

        "faiss_index":
            str(INDEX_PATH),

        "metadata":
            str(META_PATH),

        "model":
            "buffalo_s",

        "context_id":
            0,

        "detection_size":
            list(DET_SIZE),

        "faiss_vectors":
            int(index.ntotal),

        "faiss_dimension":
            int(index.d),

        "onnxruntime_version":
            ort.__version__,

        "onnxruntime_providers":
            providers,

        "validation_method":
            "leave_one_out",

        "own_vector_exclusion":
            True,

        "dataset": {

            "student_folders":
                len(student_dirs),

            "images_discovered":
                len(all_images),

            "images_scanned":
                total_images,

            "readable_images":
                readable_images,

            "read_failures":
                read_failures,

            "faces_detected":
                faces_found,

            "no_face_images":
                no_face_images,

            "multiple_face_images":
                multiple_face_images,

            "invalid_embeddings":
                invalid_embeddings,

        },

        "recognition": {

            "recognition_attempts":
                recognition_attempts,

            "correct_matches":
                correct_matches,

            "wrong_matches":
                wrong_matches,

            "recognition_accuracy_percent":
                recognition_accuracy,

            "face_detection_rate_percent":
                face_detection_rate,

            "overall_success_rate_percent":
                overall_success_rate,

            "own_vectors_excluded":
                own_vector_excluded,

            "exclusion_failures":
                exclusion_failures,

        },

        "similarity": {

            "average":
                average_similarity,

            "minimum":
                minimum_similarity,

            "maximum":
                maximum_similarity,

            "samples":
                len(similarity_scores),

        },

        "timing": {

            "model_load_sec":
                model_load_time,

            "processing_sec":
                total_elapsed
                - model_load_time,

            "total_elapsed_sec":
                total_elapsed,

            "average_inference_sec":
                average_inference,

            "inference_fps":
                inference_fps,

            "fastest_inference_sec":
                fastest_inference,

            "slowest_inference_sec":
                slowest_inference,

            "average_faiss_search_sec":
                average_faiss,

            "fastest_faiss_search_sec":
                fastest_faiss,

            "slowest_faiss_search_sec":
                slowest_faiss,

        },

        "mismatches":
            mismatches,

        "per_student":
            per_student,

        "results":
            results,

        "integrity": {

            "student_index_modified":
                False,

            "student_metadata_modified":
                False,

            "attendance_db_modified":
                False,

            "faculty_index_modified":
                False,

            "source_images_modified":
                False,

        },

    }

    # ========================================================
    # WRITE JSON
    # ========================================================

    banner("WRITING PHASE 4 RESULTS")

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    # BUILD TEXT SUMMARY
    # ========================================================

    summary_lines = []

    summary_lines.append(
        "FACETRACK — PHASE 4"
    )

    summary_lines.append(
        "LEAVE-ONE-OUT / CROSS-IMAGE "
        "GPU RECOGNITION VALIDATION"
    )

    summary_lines.append(
        "=" * 78
    )

    summary_lines.append("")

    summary_lines.append(
        "Student folders       : %d"
        % len(student_dirs)
    )

    summary_lines.append(
        "Images discovered     : %d"
        % len(all_images)
    )

    summary_lines.append(
        "Images scanned        : %d"
        % total_images
    )

    summary_lines.append(
        "Readable images       : %d"
        % readable_images
    )

    summary_lines.append(
        "Read failures         : %d"
        % read_failures
    )

    summary_lines.append(
        "Faces detected        : %d"
        % faces_found
    )

    summary_lines.append(
        "No-face images        : %d"
        % no_face_images
    )

    summary_lines.append(
        "Multiple-face images  : %d"
        % multiple_face_images
    )

    summary_lines.append(
        "Invalid embeddings    : %d"
        % invalid_embeddings
    )

    summary_lines.append(
        "Recognition attempts  : %d"
        % recognition_attempts
    )

    summary_lines.append(
        "Correct matches       : %d"
        % correct_matches
    )

    summary_lines.append(
        "Wrong matches         : %d"
        % wrong_matches
    )

    summary_lines.append(
        "Own vectors excluded  : %d"
        % own_vector_excluded
    )

    summary_lines.append(
        "Exclusion failures    : %d"
        % exclusion_failures
    )

    summary_lines.append("")

    summary_lines.append(
        "Face detection rate   : %.2f%%"
        % face_detection_rate
    )

    summary_lines.append(
        "Recognition accuracy  : %.2f%%"
        % recognition_accuracy
    )

    summary_lines.append(
        "Overall success rate  : %.2f%%"
        % overall_success_rate
    )

    summary_lines.append("")

    summary_lines.append(
        "Similarity"
    )

    summary_lines.append(
        "------------------------------------------------------------------------------"
    )

    summary_lines.append(
        "Average similarity    : %.4f"
        % average_similarity
    )

    summary_lines.append(
        "Minimum similarity    : %.4f"
        % minimum_similarity
    )

    summary_lines.append(
        "Maximum similarity    : %.4f"
        % maximum_similarity
    )

    summary_lines.append("")

    summary_lines.append(
        "Timing"
    )

    summary_lines.append(
        "------------------------------------------------------------------------------"
    )

    summary_lines.append(
        "Average inference    : %.4f sec"
        % average_inference
    )

    summary_lines.append(
        "Approx inference FPS : %.2f"
        % inference_fps
    )

    summary_lines.append(
        "Average FAISS search : %.6f sec"
        % average_faiss
    )

    summary_lines.append(
        "Processing time      : %.4f sec"
        % (
            total_elapsed
            - model_load_time
        )
    )

    summary_lines.append("")

    summary_lines.append(
        "MISMATCHES"
    )

    summary_lines.append(
        "------------------------------------------------------------------------------"
    )

    if mismatches:

        for mismatch in mismatches:

            summary_lines.append(
                "%s -> %s | score=%.4f"
                % (
                    mismatch[
                        "expected_student"
                    ],

                    mismatch[
                        "predicted_student"
                    ],

                    mismatch[
                        "similarity"
                    ],
                )
            )

    else:

        summary_lines.append(
            "None"
        )

    summary_lines.append("")

    summary_lines.append(
        "INTEGRITY"
    )

    summary_lines.append(
        "------------------------------------------------------------------------------"
    )

    summary_lines.append(
        "Student index modified : NO"
    )

    summary_lines.append(
        "Student metadata modified : NO"
    )

    summary_lines.append(
        "Attendance DB modified : NO"
    )

    summary_lines.append(
        "Faculty index modified : NO"
    )

    summary_lines.append(
        "Source images modified : NO"
    )

    summary_text = "\n".join(
        summary_lines
    ) + "\n"

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(summary_text)

    print(
        "Text summary:",
        SUMMARY_PATH,
    )

    # ========================================================
    # FINAL VERDICT
    # ========================================================

    banner("PHASE 4 VERDICT")

    if (
        total_images == 237
        and readable_images == 237
        and faces_found == 237
        and recognition_attempts == 237
        and own_vector_excluded == 237
        and exclusion_failures == 0
        and correct_matches == 237
        and wrong_matches == 0
    ):

        print(
            "PASS"
        )

        print()

        print(
            "All 237 images were successfully"
        )

        print(
            "recognized against OTHER enrolled"
        )

        print(
            "images after excluding each image's"
        )

        print(
            "own enrollment vector."
        )

    elif (
        correct_matches > 0
        and own_vector_excluded
        == recognition_attempts
    ):

        print(
            "PARTIAL PASS"
        )

        print()

        print(
            "Cross-image recognition is functioning,"
        )

        print(
            "but some images produced mismatches"
        )

        print(
            "or recognition failures."
        )

    else:

        print(
            "FAIL"
        )

        print()

        print(
            "Leave-one-out validation did not"
        )

        print(
            "complete successfully."
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
        "PHASE 4 COMPLETE"
    )

    print(
        "=" * 78
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
