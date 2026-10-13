# Adapted from adaface_source/main.py. See INTEGRATION.md.
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

class NumpySimilarityIndex:

    def __init__(self, vectors):

        self.vectors = np.asarray(
            vectors,
            dtype=np.float32
        )
        self.ntotal, self.d = self.vectors.shape

    def search(self, query, k):

        query = np.asarray(
            query,
            dtype=np.float32
        )
        similarities = query @ self.vectors.T
        ids = np.argsort(
            -similarities,
            axis=1,
            kind="stable"
        )[:, :k]
        scores = np.take_along_axis(
            similarities,
            ids,
            axis=1
        )
        return scores, ids.astype(np.int64, copy=False)


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


class OriginalRecognitionRuntime:
    """First ZIP recognition loop; only video/display/attendance I/O is adapted."""

    def __init__(self):
        self.faiss_index, self.metadata, self.students = load_database()
        if self.faiss_index.ntotal != len(self.metadata) or not self.metadata:
            raise ValueError("Original AdaFace gallery is empty or inconsistent")
        self.faiss_index = NumpySimilarityIndex(
            self.faiss_index.reconstruct_n(
                0,
                self.faiss_index.ntotal
            )
        )
        self.face_detector = YOLO(FACE_MODEL_PATH)
        self.adaface_model = load_adaface()
        self.alignment_detector = load_alignment_detector()
        self.reset()

    def reset(self):
        self.recognition_cache = {}
        self.vote_history = defaultdict(lambda: deque(maxlen=VOTE_WINDOW))
        self.confirmed_tracks = {}
        self.round_robin_index = 0
        self.frame_number = 0
        self.total_recognition_attempts = 0
        self.successful_embeddings = 0
        self.face_detection_ms = 0.0
        self.adaface_ms = 0.0
        self.last_enhancement_time = 0.0
        for tracker in getattr(getattr(self.face_detector, "predictor", None), "trackers", []):
            tracker.reset()

    def step(self, frame):
        self.frame_number += 1
        self.enhanced_this_frame = False
        detection_start = time.perf_counter()
        results = self.face_detector.track(frame, imgsz=YOLO_IMGSZ, conf=FACE_CONFIDENCE, iou=YOLO_IOU, persist=True, tracker='bytetrack.yaml', verbose=False, device='cpu')
        self.face_detection_ms = (time.perf_counter() - detection_start) * 1000.0
        tracks = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                if box.id is None:
                    continue
                track_id = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                if x2 <= x1 or y2 <= y1:
                    continue
                tracks.append((track_id, x1, y1, x2, y2))
        candidates = []
        for track_id, x1, y1, x2, y2 in tracks:
            if track_id in self.confirmed_tracks:
                continue
            cache = self.recognition_cache.get(track_id)
            if cache is None:
                last_recognition_frame = -99999
            else:
                last_recognition_frame = cache.get('frame', -99999)
            frames_since_recognition = self.frame_number - last_recognition_frame
            if frames_since_recognition >= RECOGNIZE_INTERVAL:
                candidates.append((track_id, x1, y1, x2, y2))
        selected = []
        if candidates:
            total_candidates = len(candidates)
            self.round_robin_index %= total_candidates
            recognition_count = min(MAX_RECOGNITIONS_PER_FRAME, total_candidates)
            for offset in range(recognition_count):
                candidate_index = (self.round_robin_index + offset) % total_candidates
                selected.append(candidates[candidate_index])
            self.round_robin_index = (self.round_robin_index + recognition_count) % total_candidates
        else:
            self.round_robin_index = 0
        ada_start = time.perf_counter()
        for track_id, x1, y1, x2, y2 in selected:
            self.total_recognition_attempts += 1
            face_crop = crop_face(frame, x1, y1, x2, y2)
            if face_crop is None:
                self.recognition_cache[track_id] = {'name': 'UNKNOWN', 'status': 'UNKNOWN', 'score': 0.0, 'second_score': 0.0, 'margin': 0.0, 'norm': 0.0, 'frame': self.frame_number}
                self.vote_history[track_id].append('UNKNOWN')
                continue
            recognition_crop = face_crop
            enhancer = getattr(self, "face_enhancer", None)
            can_enhance = time.perf_counter() - self.last_enhancement_time >= 5.0
            if enhancer is not None and can_enhance:
                face_height, face_width = face_crop.shape[:2]
                blur_score = cv2.Laplacian(face_crop, cv2.CV_64F).var()
                if max(face_height, face_width) < 96 or blur_score < 35.0:
                    try:
                        recognition_crop = enhancer.enhance(face_crop)
                        self.last_enhancement_time = time.perf_counter()
                    except Exception as error:
                        print("[Real-ESRGAN ERROR]", error)
            embedding, emb_norm = get_embedding(self.adaface_model, self.alignment_detector, recognition_crop)
            face_area = (x2 - x1) * (y2 - y1)
            if embedding is None:
                result_data = {'name': 'UNKNOWN', 'status': 'UNKNOWN', 'score': 0.0, 'second_score': 0.0, 'margin': 0.0, 'norm': emb_norm}
            else:
                self.successful_embeddings += 1
                result_data = classify_identity(embedding, face_area, self.faiss_index, self.metadata)
                result_data['norm'] = emb_norm
            result_data['frame'] = self.frame_number
            self.recognition_cache[track_id] = result_data
            if result_data['status'] == 'CONFIRMED':
                self.vote_history[track_id].append(result_data['name'])
            else:
                self.vote_history[track_id].append('UNKNOWN')
            vote_counts = defaultdict(int)
            for vote_name in self.vote_history[track_id]:
                if vote_name == 'UNKNOWN':
                    continue
                vote_counts[vote_name] += 1
            if vote_counts:
                winner = max(vote_counts, key=vote_counts.get)
                winner_votes = vote_counts[winner]
                if winner_votes >= MIN_VOTES:
                    self.confirmed_tracks[track_id] = winner
        self.adaface_ms = (time.perf_counter() - ada_start) * 1000.0

        faces = []
        for track_id, x1, y1, x2, y2 in tracks:
            identity = self.confirmed_tracks.get(track_id)
            cached = self.recognition_cache.get(track_id, {})
            faces.append({
                "bbox": [x1, y1, x2, y2],
                "landmark": None,
                "embedding": None,
                "confidence": 1.0,
                "employee_id": identity,
                "similarity": cached.get("score", 0.0),
                "recognized": identity is not None,
                "source_track_id": track_id,
            })
        return faces
