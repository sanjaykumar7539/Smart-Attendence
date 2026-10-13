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
# FACETRACK — PHASE 2
# GPU RECOGNITION VALIDATION
#
# Purpose:
#   Validate actual CSBS face recognition against the newly
#   created 237-vector student FAISS index.
#
# This program:
#   1. Loads buffalo_s using CUDA
#   2. Loads students.faiss
#   3. Loads students.json
#   4. Tests CSBS student images
#   5. Extracts 512-D ArcFace embeddings
#   6. Searches FAISS
#   7. Reports predicted identity and similarity
#
# IMPORTANT:
#   - Does NOT modify attendance.db
#   - Does NOT modify faculty_index.faiss
#   - Does NOT modify students.faiss
#   - This is a validation program only
# ============================================================


PROJECT_ROOT = Path.home() / "facetrack" / "facetrack_project_complete"

DATASET_ROOT = Path.home() / "facetrack" / "CSBS_students"

MODEL_ROOT = PROJECT_ROOT / "models"

INDEX_PATH = PROJECT_ROOT / "student_index" / "students.faiss"

META_PATH = PROJECT_ROOT / "student_index" / "students.json"


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
}


DET_SIZE = (640, 640)

TOP_K = 5


def banner(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():

    banner("FACETRACK — PHASE 2 GPU RECOGNITION VALIDATION")

    print("Project :", PROJECT_ROOT)
    print("Dataset :", DATASET_ROOT)
    print("Model   :", MODEL_ROOT)
    print("FAISS   :", INDEX_PATH)
    print("Meta    :", META_PATH)

    # --------------------------------------------------------
    # Basic file validation
    # --------------------------------------------------------

    required_paths = [
        DATASET_ROOT,
        MODEL_ROOT,
        INDEX_PATH,
        META_PATH,
    ]

    for path in required_paths:

        if not path.exists():

            print()
            print("ERROR: Required path does not exist:")
            print(path)

            raise SystemExit(1)

    # --------------------------------------------------------
    # ONNX Runtime provider validation
    # --------------------------------------------------------

    banner("ONNX RUNTIME PROVIDERS")

    print("ONNX Runtime:", ort.__version__)

    providers = ort.get_available_providers()

    for provider in providers:
        print("  ", provider)

    if "CUDAExecutionProvider" not in providers:

        print()
        print("ERROR: CUDAExecutionProvider is unavailable.")
        print("GPU recognition cannot be validated.")

        raise SystemExit(1)

    print()
    print("CUDAExecutionProvider: AVAILABLE")

    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    banner("LOADING STUDENT FAISS INDEX")

    index = faiss.read_index(str(INDEX_PATH))

    print("FAISS vectors :", index.ntotal)
    print("FAISS dimension:", index.d)

    if index.d != 512:

        print()
        print("ERROR: Expected 512-dimensional ArcFace embeddings.")

        raise SystemExit(1)

    if index.ntotal == 0:

        print()
        print("ERROR: FAISS index is empty.")

        raise SystemExit(1)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("Metadata entries:", len(metadata))

    if len(metadata) != index.ntotal:

        print()
        print("ERROR: FAISS and metadata counts do not match.")

        raise SystemExit(1)

    # --------------------------------------------------------
    # Load InsightFace
    # --------------------------------------------------------

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

    model_load_time = time.perf_counter() - model_start

    print()
    print("InsightFace model loaded.")
    print("Context ID       : 0")
    print("Detection size   :", DET_SIZE)
    print("Model load time  : %.3f sec" % model_load_time)

    # --------------------------------------------------------
    # Find test students
    # --------------------------------------------------------

    banner("DISCOVERING CSBS TEST IMAGES")

    student_dirs = sorted(
        d
        for d in DATASET_ROOT.iterdir()
        if d.is_dir() and d.name.upper() != "MIXED"
    )

    print("Student folders:", len(student_dirs))

    if not student_dirs:

        print()
        print("ERROR: No student folders found.")

        raise SystemExit(1)

    # --------------------------------------------------------
    # Recognition statistics
    # --------------------------------------------------------

    total_images = 0
    readable_images = 0
    face_found = 0
    correct_matches = 0

    detection_times = []
    recognition_times = []
    faiss_times = []

    # --------------------------------------------------------
    # Test first image from every student
    #
    # This gives us a clean 57-student validation pass without
    # generating excessive output.
    # --------------------------------------------------------

    banner("RUNNING 57-STUDENT RECOGNITION TEST")

    for student_number, student_dir in enumerate(
        student_dirs,
        start=1,
    ):

        images = sorted(
            p
            for p in student_dir.iterdir()
            if p.is_file()
            and p.suffix in IMAGE_EXTENSIONS
        )

        if not images:

            print(
                "[%02d/%02d] %s : NO IMAGE"
                % (
                    student_number,
                    len(student_dirs),
                    student_dir.name,
                )
            )

            continue

        # Use first image for this validation pass.
        image_path = images[0]

        total_images += 1

        image = cv2.imread(str(image_path))

        if image is None:

            print(
                "[%02d/%02d] %s : IMAGE READ FAILED"
                % (
                    student_number,
                    len(student_dirs),
                    student_dir.name,
                )
            )

            continue

        readable_images += 1

        # ----------------------------------------------------
        # Face detection + embedding
        # ----------------------------------------------------

        inference_start = time.perf_counter()

        faces = app.get(image)

        inference_time = time.perf_counter() - inference_start

        detection_times.append(inference_time)

        if not faces:

            print(
                "[%02d/%02d] %-25s -> NO FACE"
                % (
                    student_number,
                    len(student_dirs),
                    student_dir.name,
                )
            )

            continue

        face_found += 1

        # Select largest detected face.
        face = max(
            faces,
            key=lambda f:
            (f.bbox[2] - f.bbox[0])
            *
            (f.bbox[3] - f.bbox[1]),
        )

        embedding = face.normed_embedding.astype(
            np.float32
        )

        embedding = embedding.reshape(1, -1)

        # ----------------------------------------------------
        # FAISS search
        # ----------------------------------------------------

        search_start = time.perf_counter()

        similarities, vector_ids = index.search(
            embedding,
            TOP_K,
        )

        search_time = time.perf_counter() - search_start

        faiss_times.append(search_time)

        best_score = float(similarities[0][0])

        best_vector_id = int(vector_ids[0][0])

        best_metadata = metadata[best_vector_id]

        predicted_student = best_metadata["student_id"]

        expected_student = student_dir.name

        if predicted_student == expected_student:

            correct_matches += 1

            status = "CORRECT"

        else:

            status = "WRONG"

        print(
            "[%02d/%02d] %-25s -> %-25s "
            "score=%.4f  %s"
            % (
                student_number,
                len(student_dirs),
                expected_student[:25],
                predicted_student[:25],
                best_score,
                status,
            )
        )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    banner("PHASE 2 RECOGNITION SUMMARY")

    print("Student folders tested :", len(student_dirs))
    print("Images tested          :", total_images)
    print("Readable images        :", readable_images)
    print("Faces found            :", face_found)
    print("Correct identity       :", correct_matches)

    if face_found > 0:

        accuracy = (
            correct_matches / face_found
        ) * 100.0

        print(
            "Recognition accuracy   : %.2f%%"
            % accuracy
        )

    else:

        print("Recognition accuracy   : N/A")

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    banner("GPU / PIPELINE TIMING")

    if detection_times:

        avg_inference = (
            sum(detection_times)
            /
            len(detection_times)
        )

        print(
            "Average InsightFace inference : %.4f sec"
            % avg_inference
        )

        print(
            "Approx inference FPS          : %.2f"
            % (1.0 / avg_inference)
        )

        print(
            "Fastest inference             : %.4f sec"
            % min(detection_times)
        )

        print(
            "Slowest inference             : %.4f sec"
            % max(detection_times)
        )

    if faiss_times:

        avg_faiss = (
            sum(faiss_times)
            /
            len(faiss_times)
        )

        print(
            "Average FAISS search          : %.6f sec"
            % avg_faiss
        )

    # --------------------------------------------------------
    # Final verdict
    # --------------------------------------------------------

    banner("PHASE 2 VERDICT")

    if face_found == 0:

        print("FAIL")
        print("No faces were detected.")

    elif correct_matches == face_found:

        print("PASS")
        print(
            "All detected CSBS test faces matched "
            "their enrolled identities."
        )

    else:

        print("PARTIAL PASS")
        print(
            "Recognition is functioning, but some "
            "identities were mismatched."
        )

    print()
    print("Student FAISS modified : NO")
    print("Student metadata modified : NO")
    print("Attendance DB modified : NO")
    print("Faculty index modified : NO")

    print()
    print("=" * 78)
    print("PHASE 2 COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
