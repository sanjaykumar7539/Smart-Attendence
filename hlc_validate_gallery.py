from pathlib import Path
import pickle

import cv2
import numpy as np
from insightface.app import FaceAnalysis


PROJECT_ROOT = Path.home() / "facetrack" / "facetrack_project_complete"
DATASET_ROOT = Path.home() / "facetrack" / "HLC_Members"

GALLERY_FILE = PROJECT_ROOT / "hlc_reference_gallery.pkl"

MODEL_ROOT = PROJECT_ROOT / "models"


print("=" * 70)
print("HLC GALLERY VALIDATION")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load gallery
# ------------------------------------------------------------

print()
print("Loading gallery...")

if not GALLERY_FILE.exists():
    raise FileNotFoundError(
        f"Gallery not found: {GALLERY_FILE}"
    )

with open(GALLERY_FILE, "rb") as f:
    gallery = pickle.load(f)

print(f"Gallery embeddings: {len(gallery)}")

members = sorted(
    set(item["member"] for item in gallery)
)

print(f"HLC members: {len(members)}")


# ------------------------------------------------------------
# 2. Validate gallery structure
# ------------------------------------------------------------

print()
print("Checking gallery structure...")

for item in gallery:

    required = [
        "member",
        "image",
        "image_path",
        "embedding",
        "embedding_shape",
        "embedding_norm",
        "detection_score",
        "bbox",
    ]

    missing = [
        key for key in required
        if key not in item
    ]

    if missing:
        raise ValueError(
            f"Gallery item missing fields: {missing}"
        )

    embedding = np.asarray(
        item["embedding"],
        dtype=np.float32,
    )

    if embedding.shape != (512,):
        raise ValueError(
            f"Unexpected embedding shape: "
            f"{embedding.shape}"
        )

print("Gallery structure: OK")


# ------------------------------------------------------------
# 3. Check embedding normalization
# ------------------------------------------------------------

norms = []

for item in gallery:

    embedding = np.asarray(
        item["embedding"],
        dtype=np.float32,
    )

    norms.append(
        float(np.linalg.norm(embedding))
    )

print(
    "Embedding norm range: "
    f"{min(norms):.6f} - {max(norms):.6f}"
)


# ------------------------------------------------------------
# 4. Load Buffalo-S
# ------------------------------------------------------------

print()
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


# ------------------------------------------------------------
# 5. Build gallery matrix
# ------------------------------------------------------------

gallery_embeddings = np.vstack(
    [
        np.asarray(
            item["embedding"],
            dtype=np.float32,
        )
        for item in gallery
    ]
)

gallery_members = [
    item["member"]
    for item in gallery
]


# ------------------------------------------------------------
# 6. Validate one image per member
# ------------------------------------------------------------

print()
print("=" * 70)
print("CLOSED-SET RECOGNITION TEST")
print("=" * 70)

correct = 0
tested = 0
failed = 0


for member in members:

    # Select the first gallery image belonging
    # to this member.
    reference = next(
        item
        for item in gallery
        if item["member"] == member
    )

    image_path = Path(
        reference["image_path"]
    )

    print()
    print(
        f"[{tested + 1:02d}/{len(members):02d}] "
        f"{member}/{reference['image']}"
    )

    image = cv2.imread(str(image_path))

    if image is None:
        print("  ERROR: could not read image")
        failed += 1
        continue

    faces = app.get(image)

    if len(faces) != 1:
        print(
            f"  ERROR: expected 1 face, "
            f"got {len(faces)}"
        )
        failed += 1
        continue

    query = np.asarray(
        faces[0].embedding,
        dtype=np.float32,
    )

    query_norm = np.linalg.norm(query)

    if query_norm == 0:
        print("  ERROR: zero query embedding")
        failed += 1
        continue

    query = query / query_norm

    # Cosine similarity because both vectors
    # are normalized.
    similarities = gallery_embeddings @ query

    top_indices = np.argsort(
        similarities
    )[::-1][:5]

    best_index = int(top_indices[0])

    predicted_member = gallery_members[
        best_index
    ]

    best_score = float(
        similarities[best_index]
    )

    print(
        f"  Predicted : {predicted_member}"
    )

    print(
        f"  Similarity: {best_score:.6f}"
    )

    print("  Top 5:")

    for rank, idx in enumerate(
        top_indices,
        start=1,
    ):
        print(
            f"    {rank}. "
            f"{gallery_members[idx]:20s} "
            f"{float(similarities[idx]):.6f}"
        )

    tested += 1

    if predicted_member == member:
        correct += 1
        print("  RESULT    : CORRECT")
    else:
        print("  RESULT    : INCORRECT")


# ------------------------------------------------------------
# 7. Final summary
# ------------------------------------------------------------

print()
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

print(f"Members tested : {tested}")
print(f"Correct        : {correct}")
print(f"Incorrect      : {tested - correct}")
print(f"Failed         : {failed}")

if tested:
    accuracy = (
        correct / tested
    ) * 100

    print(
        f"Closed-set accuracy: "
        f"{accuracy:.2f}%"
    )

print()
print("Gallery validation finished.")
