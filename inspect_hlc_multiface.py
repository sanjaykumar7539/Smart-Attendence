from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


PROJECT_ROOT = Path("/home/iotadmin/facetrack/facetrack_project_complete")
OUTPUT = PROJECT_ROOT / "hlc_multiface_inspection.jpg"

images = [
    Path("/home/iotadmin/facetrack/HLC_Members/KARTHI/KARTHI1.jpg"),
    Path("/home/iotadmin/facetrack/HLC_Members/KARTHI/KARTHI4.jpg"),
    Path("/home/iotadmin/facetrack/HLC_Members/RAMANUJAM/RAMANUJAM14.jpg"),
]


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

panels = []

for path in images:

    img = cv2.imread(str(path))

    if img is None:
        print("FAILED:", path)
        continue

    faces = app.get(img)

    # Resize for display while preserving aspect ratio.
    max_height = 900
    scale = min(1.0, max_height / img.shape[0])

    display = cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )

    for i, face in enumerate(faces, start=1):

        x1, y1, x2, y2 = face.bbox.astype(int)

        x1 = int(x1 * scale)
        y1 = int(y1 * scale)
        x2 = int(x2 * scale)
        y2 = int(y2 * scale)

        score = float(face.det_score)

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            4,
        )

        label = f"Face {i}: {score:.3f}"

        cv2.putText(
            display,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

    title = f"{path.parent.name} / {path.name}"

    cv2.rectangle(
        display,
        (0, 0),
        (display.shape[1], 60),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        display,
        title,
        (15, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    panels.append(display)

# Put the three images vertically.
width = max(p.shape[1] for p in panels)

normalized = []

for p in panels:
    if p.shape[1] < width:
        canvas = np.zeros(
            (p.shape[0], width, 3),
            dtype=np.uint8,
        )
        canvas[:, :p.shape[1]] = p
        p = canvas

    normalized.append(p)

sheet = np.vstack(normalized)

cv2.imwrite(str(OUTPUT), sheet)

print()
print("Inspection image created:")
print(OUTPUT)
print()
print("Detected faces:")

for path in images:

    img = cv2.imread(str(path))
    faces = app.get(img)

    print(
        f"{path.parent.name}/{path.name}: "
        f"{len(faces)} faces"
    )

    for i, face in enumerate(faces, start=1):
        print(
            f"  Face {i}: "
            f"score={float(face.det_score):.4f}"
        )
