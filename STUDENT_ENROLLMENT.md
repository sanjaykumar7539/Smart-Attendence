# Student video test

Put each student's JPG, JPEG, or PNG photos in `dataset/<student name>/`.
Folder names serve as IDs and display names; keep them unique and stable.
The `MIXED` folder is excluded. Use photos containing only the intended student:
enrollment selects the largest face in each image.

Double-click `run_students.bat` to start the student dashboard. If photos are
added, removed, or changed, it rebuilds enrollment automatically on the next
launch. `main.py` also launches this student mode. Open http://localhost:8001.

Confirmed enrolled faces anywhere in the frame mark PRESENT automatically.
There is no detection line or crossing requirement. Multiple students can be
marked in the same frame. Identity confirmation remains enabled; unknown faces
cannot mark another student's attendance. Repeated sightings update the same
daily attendance row, with a 30-second per-student event cooldown.

YOLOv8m-face detects faces; InsightFace computes face embeddings and ArcFace
matches the enrolled identities. The face-only detector does not create boxes
for whole people or background objects. Its model is
`adaface_source/yolov8m-face.pt`.

After adding or changing photos, close the running server and rebuild with:

```powershell
.\.venv\Scripts\python.exe run_students.py --enroll
```

To enroll without starting the server:

```powershell
.\.venv\Scripts\python.exe run_students.py --enroll-only
```

Outputs are in `student_index`: `students.faiss`, `students.json`,
`runtime_meta.json`, `enrollment_report.json`, and `student_attendance.db`.
The report records unusable photos and students with no usable embeddings.
Existing faculty files are not changed. Enrollment rebuilds the gallery;
existing student attendance history is retained.

Each normal student-app startup clears only today's attendance and detection
rows to start the dashboard fresh. A database backup is saved first under
`student_index/backups` for `run_students.py`, or `integrated_data/backups` when
launched through `main.py`. Older dates and enrolled students are preserved.
`--enroll-only` does not clear attendance.

The dashboard reuses the application's attendance UI, so some labels still
say faculty/staff. Student records use the category `Student`. Presence mode
bypasses staff arrival/departure cutoffs and permission quotas. The legacy
morning/evening fields both reflect simple presence, not verified full-day
attendance. Recorded video detections use the current date; they are test
records, not attendance verified against the video's recording date.
Recognition accuracy must be checked against known identities in the video.

The student launcher preserves the video's native resolution for recognition.
This can improve distant-face detail but increases inference cost; if inference
is slower than the video FPS, intermediate frames may be skipped by the
latest-frame camera pipeline. Do not lower the identity thresholds just to
increase the number of marks; verify accepted identities against known students.

The dashboard shows all enrolled students for the current local date, including
students not yet marked. Live counts show detected faces, confirmed identities
in view, and unknown detections. They are separate from the unique students
marked PRESENT today. Confirmed faces have green boxes; other detected faces
have red boxes. Counts cannot include people missed by the detector or hidden
from the camera.

Use the dashboard Reset selector to clear previous dates or today's records
again while the app is running. A confirmation explains the scope. Each reset
saves a SQLite backup and preserves enrollment/photos. Students still visible
can be marked again after resetting today. Older dates remain until explicitly
cleared; they do not contribute to today's attendance count.

Video preview runs on a separate thread from recognition and displays the
latest captured frame without queuing older frames. Camera FPS measures the
preview; boxes refresh when inference finishes and expire if results stop.
Recognition and attendance still require confirmed identities. The health
endpoint reports `inference_fps` separately from preview `fps`.

Between AI results, optical flow follows image features to move the display
boxes with the video. It starts from the actual inference image to compensate
for processing delay. Unsupported or stale boxes disappear until detection
refreshes. This display tracking never confirms identities or writes attendance. The
prominent "Detected now" card counts face detections; "Attendance marked today"
still requires a confirmed enrolled identity. Neither is a guarantee that
occluded people were detected or that every person visible belongs to the
enrolled class.
