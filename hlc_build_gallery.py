from pathlib import Path
import csv
import pickle

import cv2
import numpy as np
from insightface.app import FaceAnalysis


PROJECT_ROOT = Path.home() / "facetrack" / "facetrack_project_complete"
DATASET_ROOT = Path.home() / "facetrack" / "HLC_Members"

BENCHMARK_CSV = PROJECT_ROOT / "hlc_gpu_benchmark_228.csv"
OUTPUT_FILE = PROJECT_ROOT / "hlc_reference_gallery.pkl"

MODEL_ROOT = PROJECT_ROOT / "models"


print("=" * 70)
print("HLC REFERENCE GALLERY BUILDER")
print("=" * 70)

print(f"Dataset : {DATASET_ROOT}")
print(f"Benchmark: {BENCHMARK_CSV}")
print(f"Output  : {OUTPUT_FILE}")
print()


# ------------------------------------------------------------
# 1. Load benchmark
# ------------------------------------------------------------

if not BENCHMARK_CSV.exists():
    raise FileNotFoundError(
        f"Benchmark CSV not found: {BENCHMARK_CSV}"
    )

with open(BENCHMARK_CSV, newline="") as f:
    rows = list(csv.DictReader(f))

print(f"Benchmark rows: {len(rows)}")


# ------------------------------------------------------------
# 2. Select only single-face images
# ------------------------------------------------------------

selected = []
excluded = []

for row in rows:
    face_count = int(row["face_count"])

    if face_count == 1:
        selected.append(row)
    else:
        excluded.append(row)

print(f"Single-face images selected: {len(selected)}")
print(f"Multi-face images excluded  : {len(excluded)}")
print()

if excluded:
    print("Excluded images:")
    for row in excluded:
        print(
            f"  {row['member']}/{row['image']} "
            f"({row['face_count']} faces)"
        )

print()


# ------------------------------------------------------------
# 3. Load Buffalo-S
# ------------------------------------------------------------

print("Loading Buffalo-S...")

app = FaceAnalysis(
    name="buffalo_s",
    root=str(MODEL_ROOT),
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640),
)

print("Buffalo-S ready.")
print()


# ------------------------------------------------------------
# 4. Build gallery
# ------------------------------------------------------------

gallery = []

success = 0
failed = 0
unexpected = 0

for index, row in enumerate(selected, start=1):

    member = row["member"]
    image_name = row["image"]

    image_path = DATASET_ROOT / member / image_name

    print(
        f"[{index:3d}/{len(selected)}] "
        f"{member}/{image_name}"
    )

    if not image_path.exists():
        print("  ERROR: image not found")
        failed += 1
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print("  ERROR: could not read image")
        failed += 1
        continue

    try:
        faces = app.get(image)
    except Exception as exc:
        print(f"  ERROR: face analysis failed: {exc}")
        failed += 1
        continue

    if len(faces) != 1:
        print(
            f"  WARNING: expected 1 face, "
            f"got {len(faces)}"
        )
        unexpected += 1
        continue

    face = faces[0]

    embedding = face.embedding

    if embedding is None:
        print("  ERROR: no embedding")
        failed += 1
        continue

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    # Normalize explicitly for cosine similarity.
    norm = np.linalg.norm(embedding)

    if norm == 0:
        print("  ERROR: zero embedding norm")
        failed += 1
        continue

    embedding_normalized = embedding / norm

    bbox = np.asarray(
        face.bbox,
        dtype=np.float32,
    )

    detection_score = float(face.det_score)

    gallery.append(
        {
            "member": member,
            "image": image_name,
            "image_path": str(image_path),
            "embedding": embedding_normalized,
            "embedding_shape": tuple(
                embedding_normalized.shape
            ),
            "embedding_norm": float(
                np.linalg.norm(embedding_normalized)
            ),
            "detection_score": detection_score,
            "bbox": bbox.tolist(),
        }
    )

    success += 1


# ------------------------------------------------------------
# 5. Save gallery
# ------------------------------------------------------------

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(
        gallery,
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )


# ------------------------------------------------------------
# 6. Summary
# ------------------------------------------------------------

members = sorted(
    set(item["member"] for item in gallery)
)

print()
print("=" * 70)
print("GALLERY BUILD COMPLETE")
print("=" * 70)

print(f"Benchmark images       : {len(rows)}")
print(f"Single-face candidates : {len(selected)}")
print(f"Successful embeddings  : {success}")
print(f"Failed                 : {failed}")
print(f"Unexpected face count  : {unexpected}")
print(f"HLC members represented: {len(members)}")
print(f"Gallery file           : {OUTPUT_FILE}")

print()
print("Embeddings per member:")

for member in members:
    count = sum(
        1
        for item in gallery
        if item["member"] == member
    )
    print(f"  {member:20s} {count:3d}")

print()
print("Gallery saved successfully.")
