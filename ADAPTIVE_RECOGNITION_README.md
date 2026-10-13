# FaceTrack Pro — Adaptive Long-Range Recognition

This build keeps the existing attendance/database/FastAPI architecture and replaces CAM-01's one-frame sticky match with a track-aware confirmation layer.

## Runtime path

Camera → InsightFace detection → SORT track → multi-frame buffer → quality gate → optional SR → ArcFace FAISS → optional AdaFace verification → temporal vote → CONFIRMED → sticky identity → line crossing → attendance.

## Few-shot enrollment matching

Enrollment stores one normalized ArcFace embedding per usable photo, rather than
averaging the photos into a single vector. It also writes a companion
`*_quality.json` file with a bounded image-quality weight for each vector.
At match time, each identity is scored using its quality-weighted prototype and
its best individual template. The top-two margin is calculated between
different identities, followed by the existing open-set threshold and
track-level temporal confirmation.

Student dataset enrollment writes these weights into `student_index/runtime_meta.quality.json`.
Older galleries without the companion file remain supported with neutral
weights. Existing AdaFace verification remains optional and cannot override an
ArcFace/AdaFace identity disagreement. Thresholds still require validation on
held-out camera sessions; few-shot matching alone does not guarantee higher
accuracy.

### Good face
Uses the ArcFace embedding already produced by InsightFace. No SR call.

### Medium / poor face
Buffers recent crops and uses the highest-quality sample. A small face can use the configured Real-ESRGAN-style ONNX model once per track. The enhanced crop is re-detected to obtain a fresh ArcFace embedding.

### AdaFace
AdaFace is optional and verifier-only. It never replaces ArcFace with a conflicting identity. It activates only when all three files exist:

- `models/adaface_ir101.onnx`
- `database/adaface_index.faiss`
- `database/adaface_meta.json`

Without them, the project continues in ArcFace-only mode.

### Super Resolution
No SR weights are bundled. Put a compatible Real-ESRGAN-style ONNX model at:

`models/realesrgan_x2.onnx`

If absent, SR is disabled gracefully; recognition continues.

## Important safety behavior

- Attendance is written only after `CONFIRMED`.
- `UNCERTAIN` and `UNKNOWN` never mark attendance.
- Identity becomes sticky only after temporal voting confirms it.
- FAISS top-2 margin compares different identities, which supports multiple templates per employee without rejecting two templates of the same person as "ambiguous".

## Main tuning values

Edit `config.py` → `ModelConfig`:

- `good_face_px`, `medium_face_px`
- `threshold_good`, `threshold_medium`, `threshold_poor`
- `recognition_vote_n`, `recognition_vote_window`
- `sr_trigger_face_px`, `sr_max_calls_per_track`
- `adaface_threshold`, `adaface_weight`

Start conservative. Do not lower recognition thresholds just to force distant faces to a name; preserve `UNCERTAIN/UNKNOWN` for genuinely information-poor crops.

## Run

Use the project's existing environment and commands. Install dependencies with:

`pip install -r requirements.txt`

Then run:

`python main.py`

## Added modules

- `quality.py` — size/blur/light/pose scoring
- `super_resolution.py` — optional ONNX SR fallback
- `adaface_verifier.py` — optional AdaFace + FAISS verifier
- `adaptive_recognition.py` — per-track buffering, adaptive path, fusion and temporal voting
