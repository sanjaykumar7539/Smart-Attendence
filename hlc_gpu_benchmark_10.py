from pathlib import Path
import time
import csv
import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path("/home/iotadmin/facetrack/facetrack_project_complete")
HLC_ROOT = Path("/home/iotadmin/facetrack/HLC_Members")
OUTPUT = PROJECT_ROOT / "hlc_gpu_benchmark_10.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Deterministic selection:
# first 10 images alphabetically across the HLC dataset.
images = sorted(
    p for p in HLC_ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
)[:10]

if len(images) < 10:
    raise RuntimeError(f"Only found {len(images)} images.")

print("=" * 60)
print("HLC 10-IMAGE GPU BENCHMARK")
print("=" * 60)
print(f"HLC dataset : {HLC_ROOT}")
print(f"Images      : {len(images)}")
print(f"Output      : {OUTPUT}")
print()

# ---------------------------------------------------------
# Load Buffalo-S
# ---------------------------------------------------------

print("Loading Buffalo-S...")

app = FaceAnalysis(
    name="buffalo_s",
    root=str(PROJECT_ROOT / "models"),
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
)

app.prepare(ctx_id=0, det_size=(640, 640))

print("Buffalo-S ready.")
print()

# ---------------------------------------------------------
# Benchmark
# ---------------------------------------------------------

results = []

for index, image_path in enumerate(images, start=1):

    print(f"[{index}/10] {image_path.name}")

    img = cv2.imread(str(image_path))

    if img is None:
        print("  ERROR: image could not be read")
        continue

    height, width = img.shape[:2]

    # Warm-up is already done by model initialization.
    start = time.perf_counter()

    faces = app.get(img)

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    face_count = len(faces)

    if face_count > 0:

        # Highest-confidence face
        face = max(faces, key=lambda f: float(f.det_score))

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        face_width = max(0, x2 - x1)
        face_height = max(0, y2 - y1)

        detection_score = float(face.det_score)

        embedding_shape = (
            str(face.embedding.shape)
            if face.embedding is not None
            else ""
        )

        embedding_norm = (
            float(np.linalg.norm(face.embedding))
            if face.embedding is not None
            else 0.0
        )

    else:

        x1 = y1 = x2 = y2 = 0
        face_width = face_height = 0
        detection_score = 0.0
        embedding_shape = ""
        embedding_norm = 0.0

    print(f"  Image size       : {width} x {height}")
    print(f"  Faces detected   : {face_count}")
    print(f"  Detection score  : {detection_score:.4f}")
    print(f"  Face size        : {face_width} x {face_height}")
    print(f"  Inference time   : {elapsed_ms:.2f} ms")
    print(f"  Embedding        : {embedding_shape}")
    print(f"  Embedding norm   : {embedding_norm:.4f}")
    print()

    results.append({
        "image": str(image_path.relative_to(HLC_ROOT)),
        "width": width,
        "height": height,
        "face_count": face_count,
        "detection_score": detection_score,
        "face_width": face_width,
        "face_height": face_height,
        "inference_ms": elapsed_ms,
        "embedding_shape": embedding_shape,
        "embedding_norm": embedding_norm,
    })


# ---------------------------------------------------------
# Save CSV
# ---------------------------------------------------------

fieldnames = [
    "image",
    "width",
    "height",
    "face_count",
    "detection_score",
    "face_width",
    "face_height",
    "inference_ms",
    "embedding_shape",
    "embedding_norm",
]

with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("=" * 60)
print("BENCHMARK COMPLETE")
print("=" * 60)
print(f"Results saved to: {OUTPUT}")
