# FaceTrack Pro — Always-On Setup (Task Scheduler, no third-party tools)

Scope: keeps FaceTrack Pro running unattended for a few weeks — auto-restarts
on crash, survives machine reboots, backs up the database daily, logs
rotate automatically. Uses only what's already built into Windows.

Run everything below in an **Administrator Command Prompt** (right-click
Command Prompt → "Run as administrator") — `schtasks` needs admin rights
for tasks that run whether a user is logged in or not.

---

## 1. Task: run FaceTrack Pro on boot, restart-on-crash

This runs `run_facetrack.bat`, which itself loops and restarts
`python main.py` if it ever exits — so this one task covers both
"start on boot" and "restart on crash."

```
schtasks /create /tn "FaceTrack Pro" /tr "D:\Projects\facetrack_project_hp\run_facetrack.bat" /sc onstart /ru SYSTEM /rl highest /f
```

- `/sc onstart` — triggers at machine boot (before any user logs in)
- `/ru SYSTEM` — runs as the SYSTEM account, so it works even if nobody
  ever logs into this machine
- `/rl highest` — runs with elevated rights (needed for camera/network access
  in some configurations)
- `/f` — overwrite if a task with this name already exists (safe to re-run)

**Start it right now** without waiting for a reboot:
```
schtasks /run /tn "FaceTrack Pro"
```

**Check it's running:**
```
schtasks /query /tn "FaceTrack Pro" /v /fo list
```
Look for `Status: Running`.

**Stop it** (e.g. before applying a code update):
```
schtasks /end /tn "FaceTrack Pro"
```
Note: this kills the wrapper's current loop iteration; since it's a batch
loop, you may need to also kill any lingering `python.exe` process:
```
taskkill /F /IM python.exe
```

---

## 2. Task: daily database backup

```
schtasks /create /tn "FaceTrack Backup" /tr "D:\Projects\facetrack_project_hp\backup_attendance.bat" /sc daily /st 02:00 /ru SYSTEM /f
```

Runs every night at 2:00 AM. Backups land in
`D:\Projects\facetrack_project_hp\backups\attendance_YYYY-MM-DD.db`,
and anything older than 30 days is deleted automatically by the script.

---

## 3. Verify both tasks exist

```
schtasks /query /tn "FaceTrack Pro"
schtasks /query /tn "FaceTrack Backup"
```

---

## 4. Removing a task (if you ever need to)

```
schtasks /delete /tn "FaceTrack Pro" /f
schtasks /delete /tn "FaceTrack Backup" /f
```

---

## What this setup gives you

| Problem (from earlier) | Fixed by |
|---|---|
| No auto-restart on crash | `run_facetrack.bat`'s loop — restarts 10s after any exit |
| Doesn't survive reboots | Task Scheduler `/sc onstart`, runs as SYSTEM (no login needed) |
| Log files grow forever | `main.py` now rotates logs daily, deletes anything older than 14 days |
| No backups | `backup_attendance.bat` — daily snapshot, 30-day retention |

## What this does NOT give you (out of scope for "a few weeks")
- No alerting if the camera silently stops feeding frames while the
  process is still technically "running"
- No off-machine backup (backups live on the same disk — fine for weeks,
  not a real disaster-recovery plan for years)
- SYSTEM account runs without your desktop session — if `main.py` ever
  needs to pop up a GUI window (it shouldn't, it's a headless FastAPI
  server), that won't be visible. Everything happens through the
  dashboard in a browser, so this shouldn't matter in practice.
