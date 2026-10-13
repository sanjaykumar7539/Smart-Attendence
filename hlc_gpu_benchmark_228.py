from pathlib import Path
import time
import csv
import cv2
import numpy as np
from insightface.app import FaceAnalysis


PROJECT_ROOT = Path("/home/iotadmin/facetrack/facetrack_project_complete")
HLC_ROOT = Path("/home/iotadmin/facetrack/HLC_Members")
OUTPUT = PROJECT_ROOT / "hlc_gpu_benchmark_228.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

images = sorted(
    p for p in HLC_ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
)

print("=" * 70)
print("HLC 228-IMAGE BUFFALO-S GPU BENCHMARK")
print("=" * 70)
print(f"Dataset : {HLC_ROOT}")
print(f"Images  : {len(images)}")
print(f"Output  : {OUTPUT}")
print()

if len(images) != 228:
    raise RuntimeError(
        f"Expected 228 images, but found {len(images)}"
    )

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

results = []

total_start = time.perf_counter()

for index, image_path in enumerate(images, start=1):

    img = cv2.imread(str(image_path))

    if img is None:
        print(f"[{index}/228] READ ERROR: {image_path}")
        results.append({
            "member": image_path.parent.name,
            "image": image_path.name,
            "width": 0,
            "height": 0,
            "face_count": -1,
            "detection_score": 0.0,
            "face_width": 0,
            "face_height": 0,
            "inference_ms": 0.0,
            "embedding_shape": "",
            "embedding_norm": 0.0,
            "status": "READ_ERROR",
        })
        continue

    height, width = img.shape[:2]

    start = time.perf_counter()

    faces = app.get(img)

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    face_count = len(faces)

    if face_count > 0:

        face = max(
            faces,
            key=lambda f: float(f.det_score)
        )

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

        status = "OK"

    else:

        face_width = 0
        face_height = 0
        detection_score = 0.0
        embedding_shape = ""
        embedding_norm = 0.0
        status = "NO_FACE"

    results.append({
        "member": image_path.parent.name,
        "image": image_path.name,
        "width": width,
        "height": height,
        "face_count": face_count,
        "detection_score": detection_score,
        "face_width": face_width,
        "face_height": face_height,
        "inference_ms": elapsed_ms,
        "embedding_shape": embedding_shape,
        "embedding_norm": embedding_norm,
        "status": status,
    })

    print(
        f"[{index:03d}/228] "
        f"{image_path.parent.name}/{image_path.name} | "
        f"faces={face_count} | "
        f"score={detection_score:.4f} | "
        f"face={face_width}x{face_height} | "
        f"time={elapsed_ms:.2f} ms"
    )

total_elapsed = time.perf_counter() - total_start

fieldnames = [
    "member",
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
    "status",
]

with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )
    writer.writeheader()
    writer.writerows(results)

successful = sum(
    1 for r in results
    if r["status"] == "OK"
)

no_face = sum(
    1 for r in results
    if r["status"] == "NO_FACE"
)

read_error = sum(
    1 for r in results
    if r["status"] == "READ_ERROR"
)

multi_face = sum(
    1 for r in results
    if r["face_count"] > 1
)

inference_times = [
    r["inference_ms"]
    for r in results
    if r["status"] == "OK"
]

print()
print("=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)

print(f"Total images        : {len(results)}")
print(f"Successful faces    : {successful}")
print(f"No face             : {no_face}")
print(f"Read errors         : {read_error}")
print(f"Multiple faces      : {multi_face}")

if inference_times:
    print(
        f"Average inference   : "
        f"{sum(inference_times) / len(inference_times):.2f} ms"
    )

    print(
        f"Minimum inference   : "
        f"{min(inference_times):.2f} ms"
    )

    print(
        f"Maximum inference   : "
        f"{max(inference_times):.2f} ms"
    )

print(
    f"Total benchmark time: "
    f"{total_elapsed:.2f} seconds"
)

print()
print(f"CSV saved to: {OUTPUT}")
