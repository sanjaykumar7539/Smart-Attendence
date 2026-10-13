# Student FaceTrack integration

The integrated launcher uses the classroom enrollment gallery at
`student_index/` with the matching InsightFace Buffalo_S/ArcFace recognition
model and the FaceTrack dashboard, tracking, attendance, preview and export
pipeline. It does not use the older AdaFace gallery under `adaface_source/`.

## Run

Use this project's `.venv` with the dependencies in
`requirements-integration.txt`.

From this folder:

```powershell
python main.py --check
python main.py
# Optional alternative video:
python main.py --video "D:\videos\classroom.mp4"
```

`main.py` delegates to the integrated student launcher. `run_integrated.bat`
uses this folder's `.venv` if present, otherwise `python` from PATH. Open
`http://localhost:8001` after the server starts. The default input is
`D10_20261012111238.mp4`; `--video` overrides it for a single run. Buffalo_S
remains the default. To run with Buffalo_L after enrollment and comparison, use
`python main.py --model-pack buffalo_l`; it uses a separate gallery and
attendance database.

For a fresh environment, `requirements-integration.txt` combines the dependencies:

```powershell
python -m pip install -r requirements-integration.txt
```

The required YOLO face detector and Buffalo_S ONNX models must be present at the
paths checked by `python run_integrated.py --check`.

Each normal student-video launch clears only that day's attendance and
detection rows so the dashboard starts fresh. Before clearing, the launcher
backs up the database under `integrated_data/backups/`. Enrolled students and
records from earlier dates are preserved. `--check` and enrollment-only runs
do not clear attendance.

## Compare Buffalo_L without replacing the current setup

Buffalo_L weights are not included in this workspace. Install its InsightFace
pack locally under `models/models/buffalo_l/`, including `det_10g.onnx` and
`w600k_r50.onnx`. Then build a separate gallery from the same student folders:

```powershell
python enroll_csbs_students.py --model-pack buffalo_l
python benchmark_recognition.py --check
python benchmark_recognition.py --frames 12 --stride 25 --output integrated_data\recognition_benchmark.json
```

The benchmark samples the same video positions for both packs, uses the same
face detector, quality-based thresholds, and minimum identity margin, and never
writes attendance. It reports accepted face matches and distinct matched IDs;
those counts are not proof of identity accuracy, so inspect the IDs against
known students before switching. The Buffalo_L enrollment writes only to
`student_index_buffalo_l/`, leaving the current Buffalo_S gallery and attendance
database untouched. The launcher validates each pack's model/gallery pairing.

## What was preserved

- The launcher verifies and loads the 237-template, 57-student
  `student_index/` Buffalo_S gallery, then builds the runtime roster from its
  student metadata. Gallery vectors and recognition embeddings therefore use
  the same model and feature dimensions.
- YOLOv8 Face supplies face boxes; InsightFace Buffalo_S supplies aligned
  embeddings. The recognized IDs enter the existing FaceTrack tracker and
  presence attendance path.
- In CPU mode, gallery vectors are queried with NumPy rather than FAISS search.
  This avoids a Windows OpenMP runtime conflict between FAISS and PyTorch while
  retaining exact inner-product ranking for the flat cosine-similarity gallery.
- Presence mode dispatches each recognized student once per day, and the
  database enforces one attendance row per student per day.
- Recorded video is decoded in order and paced at source FPS. The camera reader
  publishes the latest frame, so intermediate frames may be skipped if
  inference falls behind. The classroom launcher preserves the video's native
  resolution instead of resizing 2560x1440 input to 1280x720.
- On 12 sequential video frames with the current Buffalo_S model and unchanged
  strict recognition thresholds, native-resolution input produced 2 distinct
  single-frame accepted student IDs versus 1 at 1280x720. Median detected face
  size was 29.5 px versus 15 px. This small sample supports preserving pixels;
  it does not measure track-confirmed attendance accuracy or guarantee 40
  recognized students.
- Student attendance and generated roster metadata are stored in
  `integrated_data/student_attendance.db` and `integrated_data/runtime_meta.json`.
  HLC and AI modules are disabled for this classroom launcher.
- Student dashboard counts, attendance rows, and live events are limited to the
  unique enrolled student roster. Staff-only dashboard sections and unrelated
  attendance fields are hidden; this filters what is displayed without deleting
  historical database records.

## Scope of edits

`run_integrated.py` configures the existing pipeline for the student gallery,
fixed video, and a separate attendance database. `recognition.py` uses NumPy
for CPU gallery search, and `main.py` prevents repeat presence events for the
same student during one run.

## Validation and limits

```powershell
python -m unittest test_integration_contract -v
```

Tests cover CPU similarity ranking, continuous red/green boxes, and one-time
presence attendance. `--check` verifies assets and imports without starting
model inference. Recognition accuracy still depends on the students being
visible and the face crops being large and clear enough; test the actual video
before relying on attendance results.
