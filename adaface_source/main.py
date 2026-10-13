import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict, deque

import cv2
import faiss
import numpy as np
import torch

from ultralytics import YOLO

from modules.attendance import AttendanceSystem


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = "testing.mp4"

FACE_MODEL_PATH = "yolov8m-face.pt"

FAISS_PATH = "data/students.faiss"
METADATA_PATH = "data/students.json"

ADAFACE_MODEL_DIR = "adaface_model"


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# InsightFace CPU
INSIGHTFACE_CTX = -1


# ============================================================
# YOLO FACE DETECTION
# ============================================================

FACE_CONFIDENCE = 0.15
YOLO_IOU = 0.45
YOLO_IMGSZ = 1280


# ============================================================
# ADAFACE
# ============================================================

MIN_FACE_SIZE = 20
MIN_EMB_NORM = 3.0

USE_CLAHE = True


# ============================================================
# RECOGNITION THRESHOLDS
# ============================================================

CONFIRM_THRESHOLD = 0.40

UNCERTAIN_THRESHOLD = 0.32

MIN_IDENTITY_MARGIN = 0.045


# ============================================================
# FAR FACE THRESHOLDS
# ============================================================

FAR_FACE_AREA = 2500

FAR_CONFIRM_THRESHOLD = 0.37

FAR_MIN_MARGIN = 0.035


# ============================================================
# TEMPORAL VOTING
# ============================================================

VOTE_WINDOW = 5

# Testing phase:
# 2 correct recognition results are enough.
MIN_VOTES = 2


# ============================================================
# FAIR RECOGNITION SCHEDULER
# ============================================================

# Same track can be reconsidered after this many frames.
RECOGNIZE_INTERVAL = 2

# CPU:
# Process max 4 identities per frame.
MAX_RECOGNITIONS_PER_FRAME = 4


# ============================================================
# DISPLAY
# ============================================================

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720


# ============================================================
# CLAHE
# ============================================================

clahe_obj = cv2.createCLAHE(
    clipLimit=2.5,
    tileGridSize=(4, 4)
)


# ============================================================
# TRANSFORMERS COMPATIBILITY PATCH
# ============================================================

def patch_transformers():

    try:

        import transformers.modeling_utils as modeling_utils

        if getattr(
            modeling_utils.PreTrainedModel,
            "_cvlface_patched",
            False
        ):
            return

        original_getattr = (
            modeling_utils.PreTrainedModel.__getattr__
        )

        def safe_getattr(self, name):

            if name == "all_tied_weights_keys":

                return getattr(
                    self,
                    "_tied_weights_keys",
                    None
                ) or {}

            return original_getattr(
                self,
                name
            )

        modeling_utils.PreTrainedModel.__getattr__ = safe_getattr

        modeling_utils.PreTrainedModel._cvlface_patched = True

    except Exception as error:

        print(
            "[WARNING] Transformers patch:",
            error
        )


patch_transformers()


# ============================================================
# LOAD ADAFACE
# ============================================================

def load_adaface():

    from transformers import AutoModel

    model_path = Path(
        ADAFACE_MODEL_DIR
    ).absolute()

    model_file = (
        model_path
        /
        "model.safetensors"
    )

    if not model_file.exists():

        raise FileNotFoundError(
            f"AdaFace model missing: {model_path}"
        )

    print()
    print("Loading AdaFace...")

    original_directory = os.getcwd()

    try:

        os.chdir(
            str(model_path)
        )

        if str(model_path) not in sys.path:

            sys.path.insert(
                0,
                str(model_path)
            )

        model = AutoModel.from_pretrained(
            str(model_path),
            trust_remote_code=True
        )

    finally:

        os.chdir(
            original_directory
        )

    model = (
        model
        .to(DEVICE)
        .eval()
    )

    print(
        "AdaFace ready on",
        DEVICE
    )

    return model


# ============================================================
# LOAD INSIGHTFACE ALIGNMENT DETECTOR
# ============================================================

def load_alignment_detector():

    from insightface.app import FaceAnalysis

    print(
        "Loading alignment detector..."
    )

    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=[
            "detection"
        ],
        providers=[
            "CPUExecutionProvider"
        ]
    )

    app.prepare(
        ctx_id=INSIGHTFACE_CTX,
        det_size=(
            640,
            640
        )
    )

    print(
        "Alignment detector ready."
    )

    return app


# ============================================================
# IMAGE ENHANCEMENT
# ============================================================

def enhance(image):

    if not USE_CLAHE:

        return image

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(
        lab
    )

    l = clahe_obj.apply(
        l
    )

    enhanced = cv2.merge(
        [
            l,
            a,
            b
        ]
    )

    return cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )


# ============================================================
# FACE ALIGNMENT
#
# Same alignment concept used during AdaFace enrollment:
#
# Face
#   ↓
# upscale if needed
#   ↓
# CLAHE
#   ↓
# InsightFace 5 landmarks
#   ↓
# norm_crop
#   ↓
# 112 x 112
# ============================================================

def align_face(
    alignment_detector,
    bgr,
    target=112
):

    from insightface.utils import face_align

    if bgr is None:

        return None

    if bgr.size == 0:

        return None

    scales = [
        1.0,
        1.5,
        2.0,
        3.0,
        4.0
    ]

    for scale in scales:

        if scale > 1.01:

            image = cv2.resize(
                bgr,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_LANCZOS4
            )

        else:

            image = bgr.copy()

        image = enhance(
            image
        )

        try:

            faces = alignment_detector.get(
                image
            )

        except Exception:

            continue

        if not faces:

            continue

        # Largest detected face inside crop.
        face = max(
            faces,
            key=lambda f:
                (
                    f.bbox[2]
                    -
                    f.bbox[0]
                )
                *
                (
                    f.bbox[3]
                    -
                    f.bbox[1]
                )
        )

        width = (
            face.bbox[2]
            -
            face.bbox[0]
        )

        height = (
            face.bbox[3]
            -
            face.bbox[1]
        )

        if min(
            width,
            height
        ) < MIN_FACE_SIZE:

            continue

        if getattr(
            face,
            "kps",
            None
        ) is None:

            continue

        try:

            aligned = face_align.norm_crop(
                image,
                landmark=face.kps,
                image_size=target
            )

        except Exception:

            continue

        return aligned

    return None


# ============================================================
# ADAFACE INPUT TENSOR
# ============================================================

def make_tensor(image):

    tensor = torch.from_numpy(
        image.astype(
            np.float32
        )
    )

    tensor = (
        tensor
        .permute(
            2,
            0,
            1
        )
        .unsqueeze(0)
    )

    tensor = (
        tensor / 255.0
        -
        0.5
    ) / 0.5

    return tensor.to(
        DEVICE
    )


# ============================================================
# ADAFACE EMBEDDING
# ============================================================

@torch.no_grad()
def get_embedding(
    model,
    alignment_detector,
    face_crop
):

    aligned = align_face(
        alignment_detector,
        face_crop
    )

    if aligned is None:

        return (
            None,
            0.0
        )

    try:

        output = model(
            make_tensor(
                aligned
            )
        )

    except Exception as error:

        print(
            "[AdaFace ERROR]",
            error
        )

        return (
            None,
            0.0
        )

    if isinstance(
        output,
        (
            tuple,
            list
        )
    ):

        embedding = output[0]

        if len(output) > 1:

            norm = output[1]

        else:

            norm = torch.norm(
                embedding,
                dim=1
            )

    else:

        embedding = output

        norm = torch.norm(
            embedding,
            dim=1
        )

    embedding = (
        embedding
        .squeeze()
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    try:

        norm_value = float(
            norm
            .squeeze()
            .detach()
            .cpu()
            .item()
        )

    except Exception:

        norm_value = float(
            np.linalg.norm(
                embedding
            )
        )

    if norm_value < MIN_EMB_NORM:

        return (
            None,
            norm_value
        )

    vector_norm = np.linalg.norm(
        embedding
    )

    if vector_norm <= 1e-10:

        return (
            None,
            norm_value
        )

    # Cosine similarity / IndexFlatIP
    embedding = (
        embedding
        /
        vector_norm
    )

    return (
        embedding,
        norm_value
    )


# ============================================================
# CROP YOLO FACE
#
# Add context so InsightFace landmark detector can align it.
# ============================================================

def crop_face(
    frame,
    x1,
    y1,
    x2,
    y2
):

    frame_height, frame_width = (
        frame.shape[:2]
    )

    face_width = (
        x2 - x1
    )

    face_height = (
        y2 - y1
    )

    if (
        face_width <= 0
        or
        face_height <= 0
    ):

        return None

    margin_x = int(
        face_width * 0.35
    )

    margin_y = int(
        face_height * 0.35
    )

    cx1 = max(
        0,
        x1 - margin_x
    )

    cy1 = max(
        0,
        y1 - margin_y
    )

    cx2 = min(
        frame_width,
        x2 + margin_x
    )

    cy2 = min(
        frame_height,
        y2 + margin_y
    )

    if (
        cx2 <= cx1
        or
        cy2 <= cy1
    ):

        return None

    crop = frame[
        cy1:cy2,
        cx1:cx2
    ]

    if crop.size == 0:

        return None

    return crop


# ============================================================
# LOAD FAISS DATABASE
# ============================================================

def load_database():

    if not os.path.exists(
        FAISS_PATH
    ):

        raise FileNotFoundError(
            f"{FAISS_PATH} not found."
        )

    if not os.path.exists(
        METADATA_PATH
    ):

        raise FileNotFoundError(
            f"{METADATA_PATH} not found."
        )

    print()
    print(
        "Loading FAISS..."
    )

    index = faiss.read_index(
        FAISS_PATH
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(
            file
        )

    students = sorted(
        set(
            item[
                "student_name"
            ]
            for item in metadata
        )
    )

    print(
        "FAISS vectors:",
        index.ntotal
    )

    print(
        "Students:",
        len(students)
    )

    if index.ntotal != len(
        metadata
    ):

        print()
        print(
            "[WARNING]"
        )

        print(
            "FAISS vector count:",
            index.ntotal
        )

        print(
            "Metadata count:",
            len(metadata)
        )

    return (
        index,
        metadata,
        students
    )


# ============================================================
# SEARCH STUDENT
#
# Query:
#    one live embedding
#
# Database:
#    multiple embeddings per student
#
# We group returned templates by student.
# ============================================================

def search_student(
    embedding,
    index,
    metadata
):

    if embedding is None:

        return (
            "UNKNOWN",
            0.0,
            0.0,
            0.0
        )

    query = (
        embedding
        .reshape(
            1,
            -1
        )
        .astype(
            np.float32
        )
    )

    k = min(
        30,
        index.ntotal
    )

    scores, ids = index.search(
        query,
        k
    )

    per_student = defaultdict(
        list
    )

    for score, vector_id in zip(
        scores[0],
        ids[0]
    ):

        if vector_id < 0:

            continue

        if vector_id >= len(
            metadata
        ):

            continue

        student_name = metadata[
            vector_id
        ][
            "student_name"
        ]

        per_student[
            student_name
        ].append(
            float(score)
        )

    if not per_student:

        return (
            "UNKNOWN",
            0.0,
            0.0,
            0.0
        )

    student_results = []

    for (
        student_name,
        student_scores
    ) in per_student.items():

        student_scores.sort(
            reverse=True
        )

        top_scores = (
            student_scores[:3]
        )

        # ---------------------------------------------
        # Multi-template scoring
        # ---------------------------------------------

        if len(
            top_scores
        ) == 1:

            final_score = (
                top_scores[0]
            )

        elif len(
            top_scores
        ) == 2:

            final_score = (
                top_scores[0]
                * 0.65
                +
                top_scores[1]
                * 0.35
            )

        else:

            final_score = (
                top_scores[0]
                * 0.55
                +
                top_scores[1]
                * 0.30
                +
                top_scores[2]
                * 0.15
            )

        student_results.append(
            (
                student_name,
                float(
                    final_score
                )
            )
        )

    student_results.sort(
        key=lambda item:
            item[1],
        reverse=True
    )

    top1_name = (
        student_results[0][0]
    )

    top1_score = (
        student_results[0][1]
    )

    if len(
        student_results
    ) >= 2:

        top2_score = (
            student_results[1][1]
        )

    else:

        top2_score = 0.0

    margin = (
        top1_score
        -
        top2_score
    )

    return (
        top1_name,
        top1_score,
        top2_score,
        margin
    )


# ============================================================
# CLASSIFY IDENTITY
# ============================================================

def classify_identity(
    embedding,
    face_area,
    index,
    metadata
):

    (
        name,
        score,
        second_score,
        margin
    ) = search_student(
        embedding,
        index,
        metadata
    )

    # ---------------------------------------------
    # Adaptive threshold for far / small faces
    # ---------------------------------------------

    if face_area < FAR_FACE_AREA:

        confirm_threshold = (
            FAR_CONFIRM_THRESHOLD
        )

        margin_threshold = (
            FAR_MIN_MARGIN
        )

    else:

        confirm_threshold = (
            CONFIRM_THRESHOLD
        )

        margin_threshold = (
            MIN_IDENTITY_MARGIN
        )

    # ---------------------------------------------
    # Classification
    # ---------------------------------------------

    if (
        score >= confirm_threshold
        and
        margin >= margin_threshold
    ):

        status = "CONFIRMED"

    elif score >= UNCERTAIN_THRESHOLD:

        status = "UNCERTAIN"

    else:

        status = "UNKNOWN"

    return {

        "name":
            name,

        "status":
            status,

        "score":
            score,

        "second_score":
            second_score,

        "margin":
            margin
    }


# ============================================================
# LOAD DATABASE
# ============================================================

faiss_index, metadata, students = (
    load_database()
)


# ============================================================
# LOAD YOLO
# ============================================================

print()
print(
    "Loading YOLO face detector..."
)

face_detector = YOLO(
    FACE_MODEL_PATH
)

print(
    "YOLO face detector ready."
)


# ============================================================
# LOAD ADAFACE
# ============================================================

adaface_model = (
    load_adaface()
)


# ============================================================
# LOAD ALIGNMENT DETECTOR
# ============================================================

alignment_detector = (
    load_alignment_detector()
)


# ============================================================
# ATTENDANCE
# ============================================================

attendance = (
    AttendanceSystem()
)


# ============================================================
# OPEN VIDEO
# ============================================================

print()
print(
    "Opening video:",
    VIDEO_PATH
)

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    raise RuntimeError(
        f"Cannot open video: {VIDEO_PATH}"
    )


# ============================================================
# STATE
# ============================================================

recognition_cache = {}

vote_history = defaultdict(
    lambda:
        deque(
            maxlen=VOTE_WINDOW
        )
)

confirmed_tracks = {}

# Prevent same student from being counted twice.
attendance_students = set()


# ============================================================
# ROUND ROBIN STATE
# ============================================================

round_robin_index = 0


# ============================================================
# TIMING STATE
# ============================================================

frame_number = 0

previous_time = (
    time.perf_counter()
)

fps = 0.0

face_detection_ms = 0.0

adaface_ms = 0.0


# ============================================================
# DEBUG COUNTERS
# ============================================================

total_recognition_attempts = 0

successful_embeddings = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print()
        print(
            "Video completed."
        )

        break

    frame_number += 1


    # ========================================================
    # YOLO FACE DETECTION + BYTETRACK
    # ========================================================

    detection_start = (
        time.perf_counter()
    )

    results = face_detector.track(
        frame,
        imgsz=YOLO_IMGSZ,
        conf=FACE_CONFIDENCE,
        iou=YOLO_IOU,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        device="cpu"
    )

    face_detection_ms = (
        time.perf_counter()
        -
        detection_start
    ) * 1000.0


    # ========================================================
    # EXTRACT TRACKS
    # ========================================================

    tracks = []

    for result in results:

        if result.boxes is None:

            continue

        for box in result.boxes:

            if box.id is None:

                continue

            track_id = int(
                box.id.item()
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            if (
                x2 <= x1
                or
                y2 <= y1
            ):

                continue

            tracks.append(
                (
                    track_id,
                    x1,
                    y1,
                    x2,
                    y2
                )
            )


    # ========================================================
    # BUILD RECOGNITION CANDIDATES
    #
    # IMPORTANT:
    # No "largest face first".
    #
    # Every unconfirmed track can enter this list.
    # ========================================================

    candidates = []

    for (
        track_id,
        x1,
        y1,
        x2,
        y2
    ) in tracks:

        # Already permanently confirmed for this track.
        if track_id in confirmed_tracks:

            continue

        cache = recognition_cache.get(
            track_id
        )

        if cache is None:

            last_recognition_frame = (
                -99999
            )

        else:

            last_recognition_frame = (
                cache.get(
                    "frame",
                    -99999
                )
            )

        frames_since_recognition = (
            frame_number
            -
            last_recognition_frame
        )

        if (
            frames_since_recognition
            >=
            RECOGNIZE_INTERVAL
        ):

            candidates.append(
                (
                    track_id,
                    x1,
                    y1,
                    x2,
                    y2
                )
            )


    # ========================================================
    # FAIR ROUND-ROBIN SCHEDULER
    #
    # Example:
    #
    # Frame 1 -> tracks 1,2,3,4
    # Frame 2 -> tracks 5,6,7,8
    # Frame 3 -> tracks 9,10,11,12
    #
    # Instead of repeatedly selecting front-row faces.
    # ========================================================

    selected = []

    if candidates:

        total_candidates = len(
            candidates
        )

        round_robin_index %= (
            total_candidates
        )

        recognition_count = min(
            MAX_RECOGNITIONS_PER_FRAME,
            total_candidates
        )

        for offset in range(
            recognition_count
        ):

            candidate_index = (
                round_robin_index
                +
                offset
            ) % total_candidates

            selected.append(
                candidates[
                    candidate_index
                ]
            )

        round_robin_index = (
            round_robin_index
            +
            recognition_count
        ) % total_candidates

    else:

        round_robin_index = 0


    # ========================================================
    # ADAFACE RECOGNITION
    # ========================================================

    ada_start = (
        time.perf_counter()
    )

    for (
        track_id,
        x1,
        y1,
        x2,
        y2
    ) in selected:

        total_recognition_attempts += 1

        face_crop = crop_face(
            frame,
            x1,
            y1,
            x2,
            y2
        )

        # ---------------------------------------------
        # Invalid crop
        # ---------------------------------------------

        if face_crop is None:

            recognition_cache[
                track_id
            ] = {

                "name":
                    "UNKNOWN",

                "status":
                    "UNKNOWN",

                "score":
                    0.0,

                "second_score":
                    0.0,

                "margin":
                    0.0,

                "norm":
                    0.0,

                "frame":
                    frame_number
            }

            vote_history[
                track_id
            ].append(
                "UNKNOWN"
            )

            continue


        # ---------------------------------------------
        # AdaFace
        # ---------------------------------------------

        embedding, emb_norm = (
            get_embedding(
                adaface_model,
                alignment_detector,
                face_crop
            )
        )


        # ---------------------------------------------
        # Face area from original YOLO box
        # ---------------------------------------------

        face_area = (
            (x2 - x1)
            *
            (y2 - y1)
        )


        # ---------------------------------------------
        # Alignment / embedding failed
        # ---------------------------------------------

        if embedding is None:

            result_data = {

                "name":
                    "UNKNOWN",

                "status":
                    "UNKNOWN",

                "score":
                    0.0,

                "second_score":
                    0.0,

                "margin":
                    0.0,

                "norm":
                    emb_norm
            }

        else:

            successful_embeddings += 1

            result_data = (
                classify_identity(
                    embedding,
                    face_area,
                    faiss_index,
                    metadata
                )
            )

            result_data[
                "norm"
            ] = emb_norm


        result_data[
            "frame"
        ] = frame_number


        # ---------------------------------------------
        # Save latest recognition result
        # ---------------------------------------------

        recognition_cache[
            track_id
        ] = result_data


        # ====================================================
        # TEMPORAL VOTING
        # ====================================================

        if (
            result_data["status"]
            ==
            "CONFIRMED"
        ):

            vote_history[
                track_id
            ].append(
                result_data[
                    "name"
                ]
            )

        else:

            vote_history[
                track_id
            ].append(
                "UNKNOWN"
            )


        # ====================================================
        # COUNT VOTES
        # ====================================================

        vote_counts = defaultdict(
            int
        )

        for vote_name in (
            vote_history[
                track_id
            ]
        ):

            if vote_name == "UNKNOWN":

                continue

            vote_counts[
                vote_name
            ] += 1


        # ====================================================
        # CONFIRM TRACK
        # ====================================================

        if vote_counts:

            winner = max(
                vote_counts,
                key=vote_counts.get
            )

            winner_votes = (
                vote_counts[
                    winner
                ]
            )

            if (
                winner_votes
                >=
                MIN_VOTES
            ):

                confirmed_tracks[
                    track_id
                ] = winner

                # -----------------------------------------
                # One attendance per student.
                # -----------------------------------------

                if (
                    winner
                    not in
                    attendance_students
                ):

                    attendance.mark_attendance(
                        winner
                    )

                    attendance_students.add(
                        winner
                    )

                    print(
                        f"[CONFIRMED] "
                        f"Track {track_id} -> "
                        f"{winner}"
                    )


    adaface_ms = (
        time.perf_counter()
        -
        ada_start
    ) * 1000.0


    # ========================================================
    # DRAW FACE BOXES
    # ========================================================

    for (
        track_id,
        x1,
        y1,
        x2,
        y2
    ) in tracks:

        # ====================================================
        # CONFIRMED STUDENT
        # ====================================================

        if track_id in confirmed_tracks:

            name = (
                confirmed_tracks[
                    track_id
                ]
            )

            color = (
                0,
                255,
                0
            )

            label = (
                f"{name} | PRESENT"
            )

            score_text = ""


        # ====================================================
        # NOT YET CONFIRMED
        # ====================================================

        else:

            color = (
                0,
                0,
                255
            )

            info = recognition_cache.get(
                track_id
            )

            if info is None:

                label = (
                    f"ID:{track_id} CHECKING"
                )

                score_text = ""

            else:

                name = (
                    info["name"]
                )

                status = (
                    info["status"]
                )

                score = (
                    info["score"]
                )

                margin = (
                    info["margin"]
                )

                if (
                    status
                    ==
                    "CONFIRMED"
                ):

                    current_votes = (
                        list(
                            vote_history[
                                track_id
                            ]
                        )
                        .count(
                            name
                        )
                    )

                    label = (
                        f"{name} "
                        f"{current_votes}/{MIN_VOTES}"
                    )

                elif (
                    status
                    ==
                    "UNCERTAIN"
                ):

                    label = (
                        f"{name} UNCERTAIN"
                    )

                else:

                    label = (
                        "UNKNOWN"
                    )

                score_text = (
                    f"S:{score:.2f} "
                    f"M:{margin:.3f}"
                )


        # ====================================================
        # BOX
        # ====================================================

        cv2.rectangle(
            frame,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            color,
            2
        )


        # ====================================================
        # NAME / STATUS
        # ====================================================

        cv2.putText(
            frame,
            label,
            (
                x1,
                max(
                    15,
                    y1 - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # SCORE
        # ====================================================

        if score_text:

            cv2.putText(
                frame,
                score_text,
                (
                    x1,
                    min(
                        frame.shape[0] - 5,
                        y2 + 12
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                color,
                1,
                cv2.LINE_AA
            )


    # ========================================================
    # FPS
    # ========================================================

    current_time = (
        time.perf_counter()
    )

    elapsed = (
        current_time
        -
        previous_time
    )

    if elapsed > 0:

        instant_fps = (
            1.0
            /
            elapsed
        )

        # Smooth display slightly.
        if fps == 0:

            fps = instant_fps

        else:

            fps = (
                fps * 0.90
                +
                instant_fps * 0.10
            )

    previous_time = (
        current_time
    )


    # ========================================================
    # STATUS PANEL
    # ========================================================

    panel_width = 310
    panel_height = 165

    cv2.rectangle(
        frame,
        (
            0,
            0
        ),
        (
            panel_width,
            panel_height
        ),
        (
            0,
            0,
            0
        ),
        -1
    )

    panel = [

        f"FACES/TRACKS: {len(tracks)}",

        f"STUDENTS DB: {len(students)}",

        f"FAISS VECTORS: {faiss_index.ntotal}",

        f"PRESENT: {len(attendance_students)} / {len(students)}",

        f"CANDIDATES: {len(candidates)}",

        f"ADA SELECTED: {len(selected)}",

        f"FPS: {fps:.1f}",

        f"YOLO: {face_detection_ms:.0f} ms",

        f"ADAFACE: {adaface_ms:.0f} ms",

        (
            f"EMBEDDINGS: "
            f"{successful_embeddings}/"
            f"{total_recognition_attempts}"
        )
    ]

    panel_y = 16

    for text in panel:

        cv2.putText(
            frame,
            text,
            (
                5,
                panel_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (
                255,
                255,
                255
            ),
            1,
            cv2.LINE_AA
        )

        panel_y += 15


    # ========================================================
    # DISPLAY
    # ========================================================

    display = cv2.resize(
        frame,
        (
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT
        ),
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow(
        "Smart Attendance - AdaFace + FAISS",
        display
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = (
        cv2.waitKey(1)
        &
        0xFF
    )

    if key == ord("q"):

        print()
        print(
            "Stopped by user."
        )

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("SMART ATTENDANCE COMPLETED")
print("=" * 70)

print(
    "Students in database:",
    len(students)
)

print(
    "Students marked PRESENT:",
    len(attendance_students)
)

print(
    "Recognition attempts:",
    total_recognition_attempts
)

print(
    "Successful embeddings:",
    successful_embeddings
)

print()
print(
    "PRESENT STUDENTS:"
)

for student in sorted(
    attendance_students
):

    print(
        " -",
        student
    )


# ============================================================
# STUDENTS NOT CONFIRMED
# ============================================================

not_confirmed = [

    student

    for student in students

    if student
    not in attendance_students
]

print()
print(
    "NOT CONFIRMED:",
    len(not_confirmed)
)

for student in not_confirmed:

    print(
        " -",
        student
    )

print()
print("=" * 70)