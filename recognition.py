"""
FaceTrack Pro — recognition.py
RetinaFace detection + ArcFace embedding + FAISS cosine search.

Bugs fixed in this version
───────────────────────────
FIX-R1  det_thresh kwarg TypeError on InsightFace 0.7+
        Old: app.prepare(det_thresh=0.5) → TypeError
        New: set det_thresh on the detection model object after prepare()

FIX-R2  Missing re-normalisation before FAISS search
        ArcFace returns normed_embedding but dtype conversion can
        introduce tiny norm drift. Explicit L2-norm before every search.

FIX-R3  RetinaFace bbox out-of-bounds crash
        Clips x1,y1,x2,y2 to frame dimensions before any slicing.

FIX-R4  Empty / corrupt frame crash in ONNX Runtime
        Guards frame shape and channel count before calling app.get().

FIX-R5  FAISS-meta desync → IndexError on meta[idx]
        Detects desync at load time and at every search result.
"""

import cv2
import numpy as np
import faiss
import json
import os
from typing import Optional, Tuple, List, Dict
from loguru import logger
from insightface.app import FaceAnalysis

from config import config


class RecognitionEngine:
    """
    Two-stage face AI pipeline:
      frame → RetinaFace detect → ArcFace embed → FAISS cosine search
    """

    def __init__(self):
        self.app:   FaceAnalysis = None
        self.index: faiss.Index  = None
        self.meta:  List[str]    = []   # employee_id list, index-aligned with FAISS
        self._index_vectors: Optional[np.ndarray] = None
        self._template_weights: Optional[np.ndarray] = None
        self._load_models()
        self._load_index()

    # ── Model loading ──────────────────────────────────────────

    def _load_models(self):
        """
        FIX-R1: InsightFace 0.7+ does not accept det_thresh as a kwarg
        to prepare(). Setting it on the detection model object after
        prepare() is the correct approach.
        """
        logger.info(f"Loading InsightFace {config.model.model_pack}...")

        ctx_id   = config.model.gpu_id if config.model.use_gpu else -1
        session_kwargs = {}
        if not config.model.use_gpu:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = config.model.cpu_inference_threads
            options.inter_op_num_threads = 1
            options.add_session_config_entry('session.intra_op.allow_spinning', '0')
            session_kwargs = {'sess_options': options, 'providers': ['CPUExecutionProvider']}
        self.app = FaceAnalysis(
            name=config.model.model_pack,
            root=config.model.models_dir,
            allowed_modules=["detection", "recognition"],
            **session_kwargs,
        )
        # FIX-R1: do NOT pass det_thresh here — it causes TypeError on 0.7+
        self.app.prepare(ctx_id=ctx_id, det_size=config.model.det_size)

        # FIX-R1: set det_thresh on the model object after prepare
        det_model = None
        for candidate in [
            "detection", "det_10g", "det_500m",
            "scrfd_500m", "retinaface_mnet025_v2",
        ]:
            det_model = self.app.models.get(candidate)
            if det_model is not None:
                break

        if det_model is None:
            # Fallback: find any model with det_thresh attribute
            for m in self.app.models.values():
                if hasattr(m, "det_thresh"):
                    det_model = m
                    break

        if det_model is not None:
            det_model.det_thresh = config.model.detection_threshold
            logger.info(f"det_thresh set to {config.model.detection_threshold}")
        else:
            logger.warning("Could not locate detection model for threshold — using default 0.5")

        device = f"GPU:{config.model.gpu_id}" if config.model.use_gpu else "CPU"
        logger.success(f"InsightFace ready on {device}")

    # ── FAISS index ────────────────────────────────────────────

    def _load_index(self):
        idx_path  = config.database.index_path
        meta_path = config.database.meta_path

        if not os.path.exists(idx_path) or not os.path.exists(meta_path):
            logger.warning("No FAISS index found. Run: python3 enroll.py --demo")
            self.index = faiss.IndexFlatIP(512)
            self.meta  = []
            self._index_vectors = np.empty((0, 512), dtype=np.float32)
            self._template_weights = np.empty((0,), dtype=np.float32)
            return

        self.index = faiss.read_index(idx_path)
        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)

        # FIX-R5: detect and repair FAISS-meta desync at load time
        if self.index.ntotal != len(self.meta):
            logger.error(
                f"FAISS-meta desync: index has {self.index.ntotal} vectors "
                f"but meta has {len(self.meta)} entries — truncating meta to match"
            )
            self.meta = self.meta[: self.index.ntotal]

        vector_count = min(self.index.ntotal, len(self.meta))
        self._index_vectors = np.asarray(
            self.index.reconstruct_n(0, vector_count),
            dtype=np.float32,
        )
        weights_path = os.path.splitext(meta_path)[0] + ".quality.json"
        try:
            with open(weights_path, encoding="utf-8") as f:
                weights = np.asarray(json.load(f), dtype=np.float32).reshape(-1)
            if len(weights) != len(self.meta) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
                raise ValueError("quality weights do not align with gallery")
            self._template_weights = weights
        except FileNotFoundError:
            self._template_weights = np.ones(len(self.meta), dtype=np.float32)
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            logger.warning(f"Invalid gallery quality weights ({e}); using neutral weights")
            self._template_weights = np.ones(len(self.meta), dtype=np.float32)

        # Move index to GPU if available
        if config.model.use_gpu:
            try:
                if faiss.get_num_gpus() > 0:
                    res        = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, config.model.gpu_id, self.index)
                    logger.info("FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"FAISS GPU transfer failed ({e}) — running on CPU")

        logger.success(
            f"FAISS ready: {self.index.ntotal} embeddings | {len(self.meta)} faculty"
        )

    # ── Face detection ─────────────────────────────────────────

    @staticmethod
    def _preprocess_for_detection(frame: np.ndarray) -> np.ndarray:
        """Apply a very light color/contrast lift before detection.

        This keeps the underlying YOLO/InsightFace model untouched while helping
        washed-out or low-saturation footage keep more reliable face boxes.
        """
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            return frame

        proc = frame.astype(np.float32)
        proc = proc * config.model.detection_contrast_gain
        proc = np.clip(proc + config.model.detection_brightness_shift, 0, 255)

        hsv = cv2.cvtColor(proc.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * config.model.detection_saturation_gain, 0, 255)
        proc = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return proc

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        FIX-R4: Guard frame validity before ONNX Runtime call.
        FIX-R3: Clip returned bbox to frame dimensions.
        """
        # FIX-R4: reject empty or malformed frames
        if frame is None:
            return []
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        proc_frame = self._preprocess_for_detection(frame)

        try:
            faces = self.app.get(proc_frame)
        except Exception as e:
            logger.error(f"RetinaFace inference error: {e}")
            return []

        results = []
        for face in faces:
            # FIX-R3: clip to frame bounds
            x1 = int(max(0, face.bbox[0]))
            y1 = int(max(0, face.bbox[1]))
            x2 = int(min(w - 1, face.bbox[2]))
            y2 = int(min(h - 1, face.bbox[3]))

            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            if min(bw, bh) < config.model.min_face_size:
                continue
            if face.normed_embedding is None:
                continue

            results.append({
                "bbox":       [x1, y1, x2, y2],
                "landmark":   face.kps,
                "embedding":  face.normed_embedding.astype(np.float32),
                "confidence": float(face.det_score),
            })

        return results

    # ── Face identification ────────────────────────────────────

    def identify(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Compatibility wrapper around identity-level detailed search."""
        eid, score, _margin = self.identify_detailed(
            embedding, k=config.model.max_search_k,
            threshold=config.model.recognition_threshold
        )
        return eid, score


    def identify_detailed(self, embedding: np.ndarray, k: int = 8, threshold: float = None):
        """Quality-weighted prototype plus multi-template open-set matching."""
        if embedding is None or self.index.ntotal == 0 or not self.meta:
            return None, 0.0, 0.0
        query = np.asarray(embedding, dtype=np.float32).reshape(-1).copy()
        norm = np.linalg.norm(query)
        if norm < 1e-10:
            return None, 0.0, 0.0
        query = (query / norm).reshape(1, -1)
        vectors = self._index_vectors
        if vectors is None or len(vectors) != len(self.meta):
            _similarities, indices = self.index.search(query, self.index.ntotal)
            vectors = np.asarray(
                [self.index.reconstruct(int(idx)) for idx in indices[0] if idx >= 0],
                dtype=np.float32,
            )
            gallery_meta = [self.meta[int(idx)] for idx in indices[0] if idx >= 0]
            weights = np.ones(len(gallery_meta), dtype=np.float32)
        else:
            gallery_meta = self.meta
            weights = getattr(self, "_template_weights", None)
            if weights is None or len(weights) != len(gallery_meta):
                weights = np.ones(len(gallery_meta), dtype=np.float32)

        if vectors.ndim != 2 or len(vectors) != len(gallery_meta):
            return None, 0.0, 0.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        template_scores = (vectors @ query.reshape(-1)).astype(np.float32)

        grouped = {}
        for idx, eid in enumerate(gallery_meta):
            grouped.setdefault(eid, []).append(idx)
        prototype_weight = float(np.clip(
            getattr(config.model, "prototype_score_weight", 0.65), 0.0, 1.0
        ))
        use_few_shot = getattr(config.model, "few_shot_matching", True)
        by_identity = {}
        for eid, indices_for_identity in grouped.items():
            sample_indices = np.asarray(indices_for_identity, dtype=np.intp)
            scores = template_scores[sample_indices]
            best_template = float(np.max(scores))
            if use_few_shot:
                identity_weights = np.maximum(weights[sample_indices], 1e-6)
                prototype = np.sum(vectors[sample_indices] * identity_weights[:, None], axis=0)
                prototype_norm = float(np.linalg.norm(prototype))
                prototype_score = (
                    float(np.dot(query.reshape(-1), prototype / prototype_norm))
                    if prototype_norm > 1e-10 else best_template
                )
                score = prototype_weight * prototype_score + (1.0 - prototype_weight) * best_template
            else:
                score = best_template
            by_identity[eid] = score

        ranked = sorted(by_identity.items(), key=lambda x: x[1], reverse=True)
        if not ranked:
            return None, 0.0, 0.0
        eid, top1 = ranked[0]
        top2 = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = top1 - top2
        threshold = config.model.recognition_threshold if threshold is None else float(threshold)
        if top1 >= threshold and margin >= config.model.min_margin:
            return eid, top1, margin
        return None, top1, margin

    def extract_embedding_from_crop(self, crop: np.ndarray):
        """Re-detect a face in an enhanced crop and return its ArcFace embedding.
        Used only by the low-quality/SR fallback path.
        """
        if crop is None or crop.size == 0:
            return None
        try:
            faces = self.app.get(crop)
        except Exception as e:
            logger.debug(f"Enhanced crop recognition failed: {e}")
            return None
        if not faces:
            return None
        face = max(faces, key=lambda f: max(0.0, (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])))
        emb = getattr(face, 'normed_embedding', None)
        return emb.astype(np.float32) if emb is not None else None

    # ── Full frame pipeline ────────────────────────────────────

    def process_frame(self, frame: np.ndarray, annotate: bool = True) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect + identify all faces in one frame.
        Returns (face_data_list, annotated_frame).
        """
        annotated = frame.copy() if annotate else frame
        faces     = self.detect_faces(frame)
        results   = []

        for face in faces:
            x1, y1, x2, y2  = face["bbox"]
            emp_id, sim      = self.identify(face["embedding"])

            results.append({
                "bbox":        face["bbox"],
                "landmark":    face.get("landmark"),
                "embedding":   face.get("embedding"),
                "confidence":  face["confidence"],
                "employee_id": emp_id,
                "similarity":  sim,
                "recognized":  emp_id is not None,
            })

            if not annotate:
                continue
            color = (
                config.display.color_recognized
                if emp_id else config.display.color_unknown
            )
            label = emp_id if emp_id else "Unknown"
            sub   = f"{sim:.0%}" if emp_id else f"sim={sim:.2f}"

            # Draw box + label
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1
            )
            cv2.rectangle(
                annotated, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1
            )
            cv2.putText(
                annotated, label, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA,
            )
            cv2.putText(
                annotated, sub, (x1, y2 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA,
            )

        return results, annotated

    def reload_index(self):
        """Hot-reload FAISS index after new enrolments without restarting."""
        logger.info("Reloading FAISS index...")
        self._load_index()
