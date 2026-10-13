#!/usr/bin/env python3

"""
FaceTrack Phase 1 — CSBS Student Enrollment

Input:
    ~/facetrack/CSBS_students/
        <student folder>/
            *.jpg
            *.jpeg
            *.png

Output:
    ~/facetrack/facetrack_project_complete/student_index/
        students.faiss
        students.json
        enrollment_report.json

Safety:
    - Does NOT modify attendance.db
    - Does NOT modify the faculty FAISS index
    - Creates a separate student FAISS index
    - One 512-D normalized ArcFace embedding per usable image
"""

import json
import argparse
import time
from pathlib import Path

import cv2
import faiss
import numpy as np
import onnxruntime as ort

from insightface.app import FaceAnalysis
from quality import enrollment_quality_weight


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = PROJECT_ROOT / "dataset"

MODEL_ROOT = PROJECT_ROOT / "models"


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

DET_SIZE = (640, 640)

MIN_FACE_SIZE = 40


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-pack",
        choices=("buffalo_s", "buffalo_l"),
        default="buffalo_s",
        help="InsightFace pack used for both enrollment and matching",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output gallery directory (defaults to a pack-specific student_index)",
    )
    args = parser.parse_args()
    model_pack = args.model_pack
    output_root = args.output_dir or PROJECT_ROOT / (
        "student_index" if model_pack == "buffalo_s" else f"student_index_{model_pack}"
    )
    faiss_path = output_root / "students.faiss"
    meta_path = output_root / "students.json"
    report_path = output_root / "enrollment_report.json"
    model_filenames = (
        ("det_500m.onnx", "w600k_mbf.onnx")
        if model_pack == "buffalo_s"
        else ("det_10g.onnx", "w600k_r50.onnx")
    )
    missing_models = [
        MODEL_ROOT / "models" / model_pack / filename
        for filename in model_filenames
        if not (MODEL_ROOT / "models" / model_pack / filename).is_file()
    ]
    if missing_models:
        raise SystemExit(
            f"Missing {model_pack} model weights:\n"
            + "\n".join(str(path) for path in missing_models)
            + "\nInstall the InsightFace model pack locally before enrollment."
        )

    print("=" * 72)
    print("FACETRACK — PHASE 1 CSBS STUDENT ENROLLMENT")
    print("=" * 72)

    print("Dataset :", DATASET_ROOT)
    print("Model   :", model_pack)
    print("Output  :", output_root)
    print()

    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    if not DATASET_ROOT.exists():
        raise SystemExit(
            f"ERROR: Dataset does not exist:\n{DATASET_ROOT}"
        )

    # --------------------------------------------------------
    # Show ONNX Runtime providers
    # --------------------------------------------------------

    print("ONNX Runtime:", ort.__version__)

    providers = ort.get_available_providers()

    print("Available providers:")

    for provider in providers:
        print("   ", provider)

    print()

    if "CUDAExecutionProvider" not in providers:
        print(
            "WARNING: CUDAExecutionProvider is not available."
        )

    # --------------------------------------------------------
    # Find student folders
    # --------------------------------------------------------

    student_dirs = sorted(
        d
        for d in DATASET_ROOT.iterdir()
        if d.is_dir()
        and d.name.upper() != "MIXED"
    )

    if not student_dirs:
        raise SystemExit(
            "ERROR: No student folders found."
        )

    print(
        "Student folders found:",
        len(student_dirs)
    )

    print()

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load InsightFace
    # --------------------------------------------------------

    print(
        f"Loading InsightFace {model_pack}..."
    )

    app = FaceAnalysis(
        name=model_pack,
        root=str(MODEL_ROOT),
        allowed_modules=[
            "detection",
            "recognition",
        ],
    )

    # --------------------------------------------------------
    # GPU
    #
    # ctx_id=0 = NVIDIA GPU 0
    # --------------------------------------------------------

    print(
        "Initializing CPU..."
    )

    app.prepare(
        ctx_id=-1,
        det_size=DET_SIZE
    )

    print()
    print(
        "InsightFace model initialized."
    )

    print(
        "CPU enrollment processing begins."
    )

    print()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_images = 0
    readable_images = 0
    read_failures = 0

    total_faces = 0
    no_face_images = 0
    small_face_images = 0
    zero_embedding_images = 0

    usable_embeddings = 0

    embeddings = []
    metadata = []

    student_results = []

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Process students
    # --------------------------------------------------------

    for student_number, student_dir in enumerate(
        student_dirs,
        start=1
    ):

        image_paths = sorted(
            p
            for p in student_dir.iterdir()
            if p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        )

        student_embeddings = 0
        student_faces = 0
        student_failures = []

        # ----------------------------------------------------
        # Process images
        # ----------------------------------------------------

        for image_path in image_paths:

            total_images += 1

            img = cv2.imread(
                str(image_path)
            )

            if img is None:

                read_failures += 1

                student_failures.append(
                    {
                        "image": image_path.name,
                        "reason": "opencv_read_failed",
                    }
                )

                continue

            readable_images += 1

            # ------------------------------------------------
            # Face detection + recognition
            # ------------------------------------------------

            faces = app.get(img)

            total_faces += len(faces)

            if not faces:

                no_face_images += 1

                student_failures.append(
                    {
                        "image": image_path.name,
                        "reason": "no_face_detected",
                    }
                )

                continue

            # ------------------------------------------------
            # Select largest face
            # ------------------------------------------------

            face = max(
                faces,
                key=lambda f:
                    (
                        (f.bbox[2] - f.bbox[0])
                        *
                        (f.bbox[3] - f.bbox[1])
                    )
            )

            width = float(
                face.bbox[2]
                - face.bbox[0]
            )

            height = float(
                face.bbox[3]
                - face.bbox[1]
            )

            student_faces += 1

            # ------------------------------------------------
            # Minimum face size
            # ------------------------------------------------

            if min(width, height) < MIN_FACE_SIZE:

                small_face_images += 1

                student_failures.append(
                    {
                        "image": image_path.name,
                        "reason":
                            "face_below_minimum_size",
                        "width": width,
                        "height": height,
                    }
                )

                continue

            # ------------------------------------------------
            # Extract embedding
            # ------------------------------------------------

            embedding = np.asarray(
                face.normed_embedding,
                dtype=np.float32
            )

            # ------------------------------------------------
            # Defensive normalization
            #
            # IndexFlatIP + normalized vectors
            # gives cosine similarity.
            # ------------------------------------------------

            norm = np.linalg.norm(
                embedding
            )

            if norm == 0:

                zero_embedding_images += 1

                student_failures.append(
                    {
                        "image": image_path.name,
                        "reason":
                            "zero_embedding",
                    }
                )

                continue

            embedding = (
                embedding / norm
            )

            # ------------------------------------------------
            # Store embedding
            # ------------------------------------------------

            embeddings.append(
                embedding
            )

            vector_id = len(metadata)

            metadata.append(
                {
                    "vector_id": vector_id,
                    "student_id": student_dir.name,
                    "student_name": student_dir.name,
                    "source_image": str(
                        image_path
                    ),
                    "source_filename":
                        image_path.name,
                    "det_score": float(
                        face.det_score
                    ),
                    "face_width": width,
                    "face_height": height,
                    "quality_weight": enrollment_quality_weight(image, face),
                    "embedding_dimension":
                        int(
                            embedding.shape[0]
                        ),
                }
            )

            student_embeddings += 1
            usable_embeddings += 1

        # ----------------------------------------------------
        # Student summary
        # ----------------------------------------------------

        student_results.append(
            {
                "student_id":
                    student_dir.name,

                "images":
                    len(image_paths),

                "faces_detected":
                    student_faces,

                "usable_embeddings":
                    student_embeddings,

                "failures":
                    student_failures,
            }
        )

        print(
            f"[{student_number:02d}/"
            f"{len(student_dirs):02d}] "
            f"{student_dir.name}: "
            f"images={len(image_paths)}, "
            f"embeddings={student_embeddings}"
        )

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not embeddings:

        raise SystemExit(
            "ERROR: No usable embeddings generated."
        )

    # --------------------------------------------------------
    # Build matrix
    # --------------------------------------------------------

    matrix = np.vstack(
        embeddings
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Validate embedding dimension
    # --------------------------------------------------------

    embedding_dimension = int(matrix.shape[1])
    if embedding_dimension != 512:

        raise SystemExit(
            "ERROR: Expected 512-D embeddings, "
            f"got {embedding_dimension}."
        )

    # --------------------------------------------------------
    # Create separate student FAISS index
    # --------------------------------------------------------

    index = faiss.IndexFlatIP(embedding_dimension)

    index.add(
        matrix
    )

    # --------------------------------------------------------
    # Save FAISS
    # --------------------------------------------------------

    faiss.write_index(
        index,
        str(faiss_path)
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    with open(
        meta_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Build report
    # --------------------------------------------------------

    report = {

        "dataset":
            str(DATASET_ROOT),

        "model":
            model_pack,

        "det_size":
            list(DET_SIZE),

        "min_face_size":
            MIN_FACE_SIZE,

        "student_folders":
            len(student_dirs),

        "images_scanned":
            total_images,

        "readable_images":
            readable_images,

        "read_failures":
            read_failures,

        "total_face_detections":
            total_faces,

        "no_face_images":
            no_face_images,

        "small_face_images":
            small_face_images,

        "zero_embedding_images":
            zero_embedding_images,

        "usable_embeddings":
            usable_embeddings,

        "embedding_dimension":
            embedding_dimension,

        "faiss_index_type":
            "IndexFlatIP",

        "faiss_vectors":
            int(index.ntotal),

        "elapsed_seconds":
            round(
                elapsed,
                3
            ),

        "images_per_second":
            round(
                total_images / elapsed,
                3
            )
            if elapsed
            else None,

        "embeddings_per_second":
            round(
                usable_embeddings / elapsed,
                3
            )
            if elapsed
            else None,

        "database_modified":
            False,

        "faculty_index_modified":
            False,

        "student_results":
            student_results,
    }

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()

    print("=" * 72)
    print("ENROLLMENT COMPLETE")
    print("=" * 72)

    print(
        f"Student folders       : "
        f"{len(student_dirs)}"
    )

    print(
        f"Images scanned        : "
        f"{total_images}"
    )

    print(
        f"Readable images       : "
        f"{readable_images}"
    )

    print(
        f"Read failures         : "
        f"{read_failures}"
    )

    print(
        f"Face detections       : "
        f"{total_faces}"
    )

    print(
        f"Usable embeddings     : "
        f"{usable_embeddings}"
    )

    print(
        f"Embedding dimension   : "
        f"{embedding_dimension}"
    )

    print(
        f"FAISS vectors         : "
        f"{index.ntotal}"
    )

    print(
        f"Elapsed time (sec)    : "
        f"{elapsed:.3f}"
    )

    print()

    print(
        "FAISS index           :",
        faiss_path
    )

    print(
        "Metadata              :",
        meta_path
    )

    print(
        "Report                :",
        report_path
    )

    print()

    print(
        "Attendance DB changed : NO"
    )

    print(
        "Faculty index changed : NO"
    )

    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
