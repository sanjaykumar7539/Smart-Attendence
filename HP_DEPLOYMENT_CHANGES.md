# FaceTrack Pro — HP Pro Tower 280 G9 E Deployment Changes

Target: Intel i7-14700, Intel UHD 770 (no NVIDIA GPU), Windows 11 Pro, 16GB RAM.
Original target: NVIDIA Jetson Orin Nano Super, JetPack R36.4.7, Ubuntu 22.04 (aarch64).

## Files changed

### `requirements.txt`
- `onnxruntime-gpu==1.16.3` → `onnxruntime==1.16.3` (CPU build — no CUDA on this box)
- `faiss-gpu==1.7.4` → `faiss-cpu==1.7.4`
- Everything else unchanged (fastapi, opencv-python, numpy, filterpy, scipy, loguru, pydantic)

### `config.py`
- `ModelConfig.use_gpu`: `True` → `False`
  Prevents `recognition.py` from attempting `CUDAExecutionProvider` / FAISS GPU
  transfer on every startup (it would fail and fall back anyway — this just
  skips the wasted attempt and the warning-log noise).

### No changes needed
- `camera_reader.py` — already has a 4-tier fallback (NVDEC → GStreamer CPU →
  OpenCV+GStreamer → OpenCV+FFmpeg). On Windows it'll drop straight to
  OpenCV+FFmpeg since GStreamer bindings won't be present. No GStreamer
  install needed.
- `recognition.py` — already wraps GPU calls in try/except with CPU fallback;
  now goes straight to the CPU path since `use_gpu=False`.
- FastAPI, SQLite (`database.py`, `hlc_*_db.py`), audit/alert engines,
  `ai_engine_v2.py` (Claude API calls) — all pure Python, platform-agnostic.
- RTSP camera URLs — unchanged, network-based, unaffected by host OS.

## Setup on the HP box

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

- Open port 8001 in Windows Defender Firewall for LAN access to the dashboard.
- Ensure the HP box and HikVision cameras are on the same subnet.
- Run: `python main.py`
- For auto-start on boot (no systemd on Windows): wrap with Task Scheduler
  or NSSM instead of a `.service` file.

## Dropped Jetson-only constraints (not applicable here)
- `cuda-python==12.6.0` pin — no NVIDIA GPU to target
- The aarch64-specific numpy==1.26.4 rationale doesn't apply on x64;
  requirements.txt already used numpy==1.24.4, kept as-is (works fine with
  CPU onnxruntime==1.16.3 on x64)

## Performance note
`buffalo_s` (already the configured model pack) is the lightweight InsightFace
pack — the right choice for CPU-only inference. Don't switch to `buffalo_l` on
this box. The i7-14700's 20 cores should comfortably handle CPU inference for
a single/few-camera pilot; if you scale up concurrent HLC door streams, the
16GB→32GB RAM upgrade (already flagged in the HP system config doc) becomes
worth doing.

## Enrollment schema update (2026-07-23)
`faculty` table now has two new columns:
- `staff_category` — "Teaching" or "Non-Teaching" (default: Teaching)
- `is_hlc` — 1/0 flag, whether this person is also an HLC team member

Existing `attendance.db` files get these columns added automatically via a
migration in `database.py` on next startup — no manual SQL needed.

`enroll.py` now accepts:
```
python enroll.py --id EMP010 --name "Mr. Kumar" --dept "Admin Office" \
    --category Non-Teaching --images img1.jpg img2.jpg
```
- `--category Teaching|Non-Teaching` (default Teaching)
- `--hlc` — if set, also registers the person in `hlc_team.db` (via
  `HLCTeamDB.add_employee`), so they appear correctly in HLC door tracking,
  not just the main faculty/attendance system.

`add_faculty()` in `database.py` changed from `INSERT OR IGNORE` to an
upsert (`ON CONFLICT DO UPDATE`) — re-running enrollment for an existing
employee ID now updates their details instead of silently doing nothing.
Useful if you need to correct a name/dept/category after the fact, but note
the behavior change if anything elsewhere relied on the old ignore-on-conflict
semantics.

## Enrollment photo storage (2026-07-23)
Previously, enrollment photos were only read in memory to extract the face
embedding, then discarded — nothing was saved to disk. Now every enrollment
(`--images`, `--live`, or `--batch`) automatically saves the source photos
into a dedicated per-person folder:
```
faces/{employee_id}_{name}/img_01.jpg
faces/{employee_id}_{name}/img_02.jpg
...
```
This uses `config.database.faces_dir` (`./faces` by default) and matches
the folder convention `--batch` already expected. Re-enrolling the same
person appends new photos with incrementing filenames rather than
overwriting — nothing is lost.

## Windows UTF-8 encoding fix (2026-07-23)
On Linux, `open()` defaults to UTF-8. On Windows, it defaults to the system
code page (cp1252 on this box), which crashed on non-ASCII characters in
`dashboard.html` (`UnicodeDecodeError` on `GET /`). Fixed by adding
`encoding="utf-8"` explicitly to every text-mode `open()` call:
- `main.py` — `dashboard()` and `hr_panel()` routes
- `enroll.py` — FAISS meta JSON read/write
- `recognition.py` — FAISS meta JSON read
No other `open()` calls in the codebase were missing this.

## Batch enrollment with mixed staff types (2026-07-23)
`enroll.py --batch` now supports an optional `roster.csv` inside the batch
folder to give each person their own department/category/HLC status —
needed for real batches that mix Teaching, Non-Teaching, and HLC staff.

Folder layout:
```
enroll_batch/
  roster.csv
  FAC001_Dr_Ramesh_Kumar/
    img1.jpg
    img2.jpg
  EMP010_Mr_Kumar/
    img1.jpg
```

roster.csv format (header row required):
```
employee_id,name,department,category,is_hlc
FAC001,Dr. Ramesh Kumar,Computer Science,Teaching,yes
EMP010,Mr. Kumar,Admin Office,Non-Teaching,no
```

If `roster.csv` is absent, every person in the batch gets the same
`--category`/`--hlc` from the command line (default: Teaching, not HLC).

## AI model string updated (2026-07-23)
`ai_engine_v2.py` had `CLAUDE_MODEL = "claude-sonnet-4-5"` hardcoded (stale).
Updated to `claude-sonnet-5`, and synced the `/api/ai/status` endpoint to
report the same model it actually calls.

## EGS-branded dashboard swap (2026-07-23)
Replaced `static/dashboard.html` (was a generic dark/cyberpunk design) with
the maroon/gold EGS Pillay Engineering College branded version — college
banner, crest, NAAC A++/NBA badges, "Dept. of AI & Data Science" sub-brand.

The new dashboard was built against a slightly different HR API shape than
what's actually running, so these fixes were needed:

**Path/payload fixes (done — no policy decisions needed):**
- Leave submission: `/api/hr/leave` → `/hr/leave/apply`, plus auto-computed
  `days` field from the date range (backend requires it, the form didn't
  collect it)
- Leave list: `/api/hr/leaves` → new `/hr/leaves` route (flat array of
  pending leave applications)
- Approve/reject: path order fixed (`/hr/leave/approve/{id}`,
  `/hr/leave/reject/{id}`), both now send required request bodies.
  Reject now prompts for a rejection reason (backend requires one, the
  original dashboard JS didn't collect it).
- Balance lookup: new `/hr/balance/{employee_id}` route added — reshapes
  `hr_db.get_leave_balance()`'s nested-by-type dict into the flat
  `cl_remaining`/`ml_remaining`/`el_remaining` shape the dashboard expects.
- Attendance override: `/api/hr/override` → `/hr/override`, and the JSON key
  `override_status` renamed to `new_status` to match `OverrideRequest`.
- HLC force-check: `/api/hlc/alerts/force-check` → `/api/hlc/alert/force_check`.
- College Events widget: added `GET /api/college-events`, backed by an
  editable `college_events.json` in the project root (empty list if the
  file's missing — widget just shows nothing, no error). Starter file with
  two sample entries included — edit freely.

**Left unfixed — needs your input, not a guess:**
- Permissions widget (`/api/hr/permissions?month=`) — there's no concept of
  "permission" (short/half-day leave separate from formal leave types) in
  `hr_database.py`'s current schema (`LEAVE_LABELS` only has CL/ML/EL/MAT/
  PAT/COMP/OD/LOP/SL). Building this needs you to define what a
  "permission" is at EGS Pillay (a leave sub-type? a separate short-leave
  slip system?) before I model it. Until then this widget shows "Permissions
  API not available" — degrades gracefully, doesn't break anything else.
- The `perm_this_month` field in the Balance widget is hardcoded to 0 for
  the same reason.

## Dashboard tab-switching bug fixed (2026-07-23)
Tabs (Live/Attendance/HLC Alerts/AI Chat) weren't toggling on click. Root
cause: leftover/broken code in the WebSocket `onmessage` handler —
```js
if (data.name && data.name !== "Unknown") {
  status = "PRESENT";
```
had an unclosed brace (and referenced an undefined `data` variable), which
made the entire inline `<script>` block fail to parse. A JS syntax error
anywhere in a `<script>` tag silently kills the whole script — so
`switchView()` and every other `onclick` handler on the page were
non-functional, not just the tabs. Removed the dead code; verified the full
script now parses cleanly with `node --check`.

## Three showstoppers fixed (2026-07-23)

**1. HLC camera config bug** — `build_registry_from_config()` read
`config.hlc_cameras` (doesn't exist) instead of `config.hlc.cameras` (the
real location). Silently returned `[]` via `getattr` default — matches the
`HLC cameras registered: []` seen in every startup log so far. One-line fix
in `hlc_camera_config.py`.

**2. attendance.py rewired to stop duplicating the policy engine** —
`AttendanceProcessor` called four `Database` methods that never existed
(`update_last_seen`, `create_attendance`, `update_final_status`,
`count_permissions_this_month`) — would have crashed on the first real
detection. It also reimplemented the entire on-time/permission/half-day/
absent policy matrix a second time (`AttendancePolicy` class), duplicating
what `database.py`'s `log_detection()`/`_update_attendance()`/
`_derive_final()` already does correctly (including the FIX-DB1
double-counting fix).
Fix: removed the duplicate `AttendancePolicy` class entirely.
`AttendanceProcessor` now only handles multi-frame vote confirmation and
name caching (its actual job), then delegates every detection — IN or OUT —
to `db.log_detection()`, which re-derives first_seen/last_seen/morning/
evening/final status from all of today's detections in one call. Single
source of truth, one policy engine.

**3. AI chat API contract mismatch fixed** — `ai_router` was mounted with
no prefix (`app.include_router(ai_router)`), so `/chat` etc. lived at the
bare path, not `/api/ai/chat` as the dashboard called and the code's own
comments claimed. The old `/chat` route also expected `{messages:[...]}`
and streamed via SSE — but the dashboard's `sendMessage()` was already
sending `{message, date}` and expecting one JSON `{answer}` back (no
streaming). Went with the simpler, already-matching direction:
  - `main.py`: `app.include_router(ai_router, prefix="/api/ai")` — now
    actually mounted where the comment always claimed.
  - `ai_routes_v2.py`: `/chat` rewritten to accept `{message, date}` and
    return `{answer}` synchronously (collects the streamed response
    server-side via a new `chat_sync()` method rather than SSE). Added
    `/suggestions` (dashboard already had a hardcoded fallback for this;
    now backed by a real endpoint with the same list, centrally editable).
  - `ai_engine_v2.py`: `_ctx_chatbot()` now takes an optional `work_date`
    so the dashboard's date picker actually does something instead of
    always defaulting to today.
  - `/dataset`, `/analytics`, `/report`, `/debug` untouched — still SSE,
    still work as before, just correctly reachable at `/api/ai/*` now.

## Real EGSPEC brand colors applied (2026-07-23)
User provided a screenshot of egspec.org. Sampled exact pixel colors and
replaced the earlier placeholder maroon/gold palette:
- Navy `#001B70` (was `#7B1C2C` maroon) — top bar, logo box, banner primary
- Magenta `#AF005F` (was `#C9922A` gold) — nav links, secondary accent
- Yellow `#FFCC00` — new variable added, matches EGSPEC's real CTA/badge
  color (TNEA code badge, admission-open callouts)

All changed via the `:root` CSS variables (35 usages across the file update
automatically), plus the hardcoded college crest SVG and one JS color
reference that used raw hex instead of the variables.

## Always-on setup for a few weeks of unattended operation (2026-07-23)
Three new files, no third-party tools required (uses Windows' built-in
Task Scheduler + loguru's built-in rotation, already a dependency):

- **`main.py`** — now configures loguru to write to `logs/facetrack_YYYY-MM-DD.log`
  with daily rotation and 14-day auto-retention. Previously logged to
  console only; closing the terminal lost everything, and nothing rotated.
- **`run_facetrack.bat`** (new) — restart-on-crash wrapper. Loops
  `python main.py`; if it exits for any reason, waits 10s and restarts.
- **`backup_attendance.bat`** (new) — daily `attendance.db` snapshot to
  `backups/`, 30-day retention, auto-deletes older backups.
- **`TASK_SCHEDULER_SETUP.md`** (new) — exact `schtasks` commands to run
  both scripts unattended: FaceTrack Pro on boot (as SYSTEM, survives
  reboots without anyone logged in) + nightly backup at 2 AM.

This covers items 1, 3, 4 from the "genuine 24/7 operation" list (auto-restart,
DB backup, log rotation). Camera reconnect (item 2) was already handled by
`camera_reader.py`. Health monitoring / alerting and off-machine backup are
explicitly out of scope for a multi-week run — see TASK_SCHEDULER_SETUP.md's
"what this does NOT give you" section if that becomes relevant later.

## Attendance policy rewritten to real EGS Pillay schedule (2026-07-24)
User-specified policy: college day 09:10-16:10 (full day), 45min permission
grace on either the morning arrival or evening departure (2/month, shared
quota), half-day allowed in either direction (forenoon-only OR
afternoon-only), forenoon session ends 12:35, afternoon session starts 13:20.

**`config.py`** — six cutoffs updated:
| Cutoff | Old | New |
|---|---|---|
| on_time_arrival_cutoff | 09:05 | **09:10** |
| permission_arrival_end | 09:55 | 09:55 (unchanged, now correctly 09:10+45min) |
| afternoon_absent_cutoff (forenoon session end) | 12:45 | **12:35** |
| on_time_departure_start | 16:10 | 16:10 (unchanged) |
| permission_departure_start | 15:25 | 15:25 (unchanged, 16:10-45min) |
| half_day_cutoff (afternoon session start) | 12:45 | **13:20** |

**`database.py`** — two real logic gaps fixed, not just cutoff values:
1. Boundary fix: arriving exactly at 09:10 now counts as on-time (`<=`
   instead of `<` — matches "college timing is FROM 9:10am").
2. `_derive_final()` rewritten. The old matrix only ever produced a
   forenoon-side half day and only ever tracked permission quota for late
   arrivals — an afternoon-only half day (missed forenoon, attended
   afternoon fully) fell through to `ABSENT_MORNING` regardless of
   afternoon attendance, and an early departure using the 45min grace
   consumed zero quota if the morning was on-time. Both fixed:
   - Half-day now resolves symmetrically — missing either full session
     (not just the morning) with the other session properly attended
     yields `HALF_DAY`.
   - Permission quota now consumes a credit any time the 45min grace is
     used on **either** side, not just arrival. Both sides using grace on
     the same day still only costs one credit.
   - Where the old code had an unconditional leniency not covered by the
     user's stated rules (on-time morning + an afternoon departure that's
     past the half-day cutoff but not within the evening grace window),
     that existing behavior (`FULL_DAY`) was deliberately preserved rather
     than tightened, since it wasn't part of what was asked to change.
   Full matrix verified against 13 scenarios plus a quota-exhaustion case
   before shipping — see chat history for the test harness if this needs
   revisiting.

## HLC dual-camera pipeline built (2026-07-24)
Confirmed the HLC camera registry/processor/alert-engine were fully built
but never actually connected to live video — `CameraReader` was hardcoded
single-instance (reads `config.camera.*` globally, no way to point it at a
second RTSP URL), and `/stream/hlc*` routes returned placeholder JSON, not
real video. This is why only one camera ever appeared anywhere.

Topology confirmed: single door, two cameras facing each other (inside +
outside). `RoomStateManager` already had dedup logic built specifically for
"same person, same direction, different camera within 5s" — the
architecture was designed for this, just never wired to real capture.

- **`camera_reader.py`** — `CameraReader` now accepts optional
  `rtsp_url`/`camera_id`/etc. constructor args, defaulting to the main
  camera's config for 100% backward compatibility. Supports multiple
  independent instances now.
- **`config.py`** — HLC camera list reconfigured for single-door/two-camera
  topology: both entries share `door_id: "DOOR_MAIN"`, opposite
  `in_direction` (they face each other). **RTSP URLs are still
  placeholders** (`192.168.0.101`/`.102`) — replace with real camera URLs
  when available. `in_direction` may need flipping after first live test —
  comment in config.py explains how to tell.
- **`main.py`** — new `hlc_processing_loop()` (mirrors the main camera's
  loop: recognition, tracking, line-crossing, dispatch — writes to
  `state.hlc_frames[camera_id]` instead of the shared main-camera buffer).
  `main()` now starts one `CameraReader` + one processing thread per
  enabled HLC camera. `/stream/hlc` and `/stream/hlc/{camera_id}` now
  serve real MJPEG video instead of placeholder JSON. New
  `/api/hlc/cameras` endpoint lists registered cameras for the dashboard.
- **`static/dashboard.html`** — HLC tab now shows live video tiles (one
  per registered camera), rendered dynamically from `/api/hlc/cameras` —
  no more alerts-table-only view.

**Not yet tested against real hardware** — RTSP URLs are placeholders.
Once real URLs are set in `config.py`, watch the startup log for both
`CameraReader[HLC-CAM-OUTSIDE]` and `CameraReader[HLC-CAM-INSIDE]`
connecting successfully, and check the HLC tab shows both video tiles live.

**Performance note:** this now runs 3 concurrent recognition loops (main +
2 HLC) on CPU. The i7-14700's 20 cores should handle it, but watch CPU load
once real cameras are connected — if it's too heavy, `frame_skip` in
`config.camera` can be raised to reduce inference frequency per camera.

## Face recognition false-match fix (2026-07-24)
Root cause: the FAISS matcher only ever checked the single best match
(`k=1`), with no check on how close the runner-up was — so two enrolled
people with even moderately similar embeddings could cause confident wrong
matches, especially combined with `recognition_threshold` having been
lowered to 0.22 (from 0.28) to help with backlit/distant faces.

- `recognition_threshold`: 0.22 → **0.30** (still more permissive than the
  original 0.28 for backlit/distant faces, but meaningfully stricter)
- New `min_margin: 0.06` — FAISS search now retrieves top-2 candidates;
  if the top-1 and top-2 similarity scores are within 0.06 of each other,
  the match is rejected as ambiguous rather than confidently guessed. This
  is what actually catches lookalike/similar-embedding false matches — the
  threshold alone can't distinguish "confidently correct" from "barely won
  against a similar person."

If mismatches continue after this, the next lever is enrollment quality —
more varied photos per person (different angles/lighting) sharpens
embeddings and increases the natural margin between different people.

## Real HLC camera RTSP URLs set (2026-07-24)
- Outside (HLC-CAM-OUTSIDE): 192.168.1.102, sub-stream (channel 102)
- Inside (HLC-CAM-INSIDE): 192.168.1.101, sub-stream (channel 102)
- Password `Admin@123` percent-encoded as `Admin%40123` (the `@` must be
  encoded — it's the RTSP URL's credential/host delimiter, an unencoded `@`
  in the password breaks URL parsing)
- Used sub-stream (channel 102, lower resolution) rather than main stream
  (101) to reduce CPU load — 3 concurrent recognition pipelines now run on
  this box (main camera + 2 HLC), and HLC only needs enough resolution to
  recognize a face crossing a doorway.

## Camera connectivity resolved (2026-07-24)
Root cause was NOT network reachability (ping confirmed both cameras
reachable) — it was the sub-stream channel. Channel 102 (sub-stream)
either isn't enabled or doesn't exist on these cameras; channel **101**
(main stream) is what actually connects. All three camera entries
(main + both HLC) switched from `/Streaming/Channels/102` to `/101`.

Main attendance camera now points at the same physical camera as
HLC-CAM-OUTSIDE (192.168.1.102) — two separate simultaneous RTSP
connections to the same camera, both on the heavier main stream (not
sub-stream, since 102 doesn't work). Worth watching after restart: if
the "Live" tab feed is laggy or drops while the HLC tab stays smooth,
that's a sign this camera's encoder is straining under 2 concurrent
full-resolution connections — the fix then would be sharing one
connection's frames between both processing loops instead of opening two,
not something to worry about unless it actually shows up.

## Fast-walker recognition fix (2026-07-24)
Real root cause (not what was initially suspected): `AttendanceTrigger.check()`
required a track to be BOTH recognized AND crossing the line on the exact
same processed frame. A fast walker crossing the doorway before recognition
converges means that crossing is lost permanently — by the time recognition
succeeds a frame or two later, they're already past the line, no new
crossing transition exists to detect. This is a timing bug, not a
confirmation-strictness problem — requiring MORE matching frames to trust
identity (as originally proposed) would have made this worse, since fast
walkers already struggle to get even one clean recognized frame before
crossing.

Fix: `AttendanceTrigger` now remembers a crossing the moment it's seen
(direction + timestamp), regardless of recognition status, and dispatches
it as soon as recognition catches up on a later frame — instead of
requiring both to align on one frame. Bounded to an 8-second wait
(`_pending_timeout_sec`) so a crossing that genuinely never gets recognized
doesn't sit in memory forever over weeks of continuous operation.

Supporting change: `frame_skip` 4 → 2 (more recognition attempts/sec within
that window). Watch CPU load after this — now 3 concurrent camera loops
(main + 2 HLC) each running inference twice as often. If the box struggles,
raise it back toward 3-4; the trigger fix is what actually matters most,
frame_skip is a secondary lever.

## HLC line-crossing removed, replaced with door-state logic (2026-07-24)
Per user request: HLC no longer uses boundary/line-crossing geometry at
all. With one camera outside the door and one inside, the camera that
recognized the person now directly IS the direction signal:
- Recognized on the INSIDE camera while not currently marked inside -> entered
- Recognized on the OUTSIDE camera while currently marked inside -> exited
- Any other combination -> no event (e.g. seen outside while already
  outside — haven't gone through the door yet)

New `HLCDoorTrigger` class replaces `AttendanceTrigger`/line-crossing for
the HLC pipeline specifically. The main camera's attendance logic is
untouched — still uses line-crossing (`AttendanceTrigger`), since this
request was scoped to HLC only.

Each track fires at most once per lifetime (`_fired` set, cleaned up each
frame against currently-active track IDs) so someone lingering in view of
one camera doesn't spam repeat events.

`main()` now derives `camera_role` ("inside"/"outside") from each camera's
ID when starting its processing thread. Confirmed compatible end-to-end
with the existing `HLCProcessorV4`/`resolve_room_direction()` — it already
had legacy passthrough support for direction values already resolved to
"IN"/"OUT" directly (rather than raw frame-crossing directions), so no
downstream changes were needed.

Bug caught before shipping: `HLCProcessorV4`'s room-state manager is
exposed as `.room`, not `.room_state` — used the wrong attribute name on
first pass, would have crashed on the first recognized detection. Fixed
before packaging.

## HLC re-scoped: unauthorized overstay flagging, not time-of-arrival attendance (2026-07-24)
Confirmed scope before touching anything: this change applies to the HLC
room specifically (unauthorized visitor detection), NOT the main staff
attendance system (on-time/permission/half-day policy) — that's untouched
and still fully intact.

Good news: the architecture already did almost exactly what was asked.
`HLCProcessorV4.process_detection()` already routes anyone who isn't a
registered HLC team member (`tdb.is_hlc_employee()`) into a separate
visitor-tracking path, and `hlc_alert_engine.py` already only ever checks
visitor sessions for overstay — HLC team members are never flagged,
regardless of how long they stay. So "flag anyone except HLC members who
overstays" was already the real rule; it just needed the threshold changed
and the framing clarified.

- `config.py`: `alert_threshold_mins` 15.0 → **5.0**
- `hlc_visitor_db_v4.py`: was hardcoding `> 15` directly in the overstay
  calculation (separate from the configurable threshold used everywhere
  else) — now correctly reads `config.hlc.alert_threshold_mins`. Also
  synced the `sessions_needing_alert()` default parameter. Note: the
  `over_15min` field/column name is kept as-is (legacy name) rather than
  renamed — renaming would need a schema migration and doesn't change
  behavior, only the logic mattered.
- `static/dashboard.html`: both HLC alert widgets (Live tab's compact
  panel, and the full HLC tab) updated from "HLC Overstay Alerts /
  Faculty inside campus for more than 15 minutes" to "Unauthorized
  Overstay Alerts / Non-HLC-member present in the room for more than 5
  minutes — auto-detected. HLC team members are never flagged." The
  dynamic threshold label (`hlc-threshold-label`) already pulls live from
  the backend, so it auto-reflects 5 minutes without needing a separate fix.

## AI chatbot switched from Anthropic to Google Gemini (2026-07-28)
User is cost-conscious and chose Gemini for its genuine recurring free
tier (no card, no expiry) over Anthropic's one-time ~$5 trial credit.

Model: `gemini-2.5-flash` (confirmed current/stable, not deprecated, as
of this change).

**`ai_engine_v2.py`** — the real rewrite. `stream()` completely rebuilt:
- Auth: Gemini takes the key as a URL query param (`?key=...`), not a
  header like Anthropic's `x-api-key`
- Request shape: Gemini uses `contents: [{role, parts: [{text}]}]` instead
  of Anthropic's `messages: [{role, content}]` — role `"assistant"` maps
  to Gemini's `"model"`
- System prompt: separate `systemInstruction` field instead of Anthropic's
  top-level `system` string
- Response parsing: Gemini's SSE stream sends `candidates[0].content.parts[]`
  chunks, not Anthropic's `content_block_delta` events
- All higher-level methods (`chatbot`, `chat_sync`, `analytics`, `dataset`,
  `report`, `debug`) untouched — they all just call `self.stream(...)`,
  so the swap was isolated to one method

**`config.py`, `main.py`, `ai_routes_v2.py`** — every `ANTHROPIC_API_KEY`
reference (env var name, comments, error messages, docstrings) updated to
`GEMINI_API_KEY`. `/api/ai/status` now reports `gemini-2.5-flash`.

**Action needed:** update `.env` —
```
GEMINI_API_KEY=AIza...
```
(remove or leave `ANTHROPIC_API_KEY=` — it's no longer read). Get a free
key at https://aistudio.google.com/apikey (Google account, no card).

## Sandbox reset recovery + Ubuntu Linux port (2026-08-10)
User switched their machine's OS from Windows to Ubuntu Linux and lost all
local files in the process. Rebuilt the complete project from the last
successfully delivered zip, re-applied the `Optional` import fix (made
just before an unrelated tooling reset wiped the working copy before that
fix was repackaged — confirmed via pyflakes this was the only casualty,
full undefined-name sweep across all files came back clean).

Added `SETUP_LINUX.md` — complete Ubuntu setup guide covering: Python 3.11
via deadsnakes PPA (Ubuntu doesn't ship 3.11 by default on 22.04 or
24.04), required system packages (libgl1, ffmpeg, build-essential),
optional real GStreamer support (Linux can actually use it, unlike
Windows which always fell back to FFmpeg), and systemd services/timers
replacing Task Scheduler + the .bat wrapper entirely — systemd's
`Restart=always` covers crash recovery and boot-start natively, no
polling wrapper script needed like the Windows `run_facetrack.bat` loop.

requirements.txt is unchanged and works as-is on Linux — no Windows-only
package pins existed (loguru's win32-setctime dependency is handled
automatically via pip environment markers, no manual action needed).
config.py's camera RTSP URLs and all other settings carry over unchanged
— same physical hardware/cameras, just a different OS on the compute box.
