# FaceTrack Pro — Ubuntu Linux Setup Guide

Same HP Pro Tower 280 G9 E hardware (Intel i7-14700, Intel UHD 770, no
NVIDIA GPU) — just Windows swapped for Ubuntu. Everything below assumes a
fresh Ubuntu install (22.04, 24.04, or 24.10 — instructions work on all
three).

---

## 1. Exact Python version needed: 3.11

Same constraint as on Windows: `onnxruntime==1.16.3`, `numpy==1.24.4`,
`faiss-cpu==1.7.4`, and `insightface==0.7.3` all need Python 3.11
specifically — no prebuilt wheels exist for 3.12/3.13/3.14 at these
versions, and building from source is a much bigger headache than just
installing 3.11 alongside whatever Ubuntu ships by default.

Ubuntu doesn't ship 3.11 by default on 24.04 (ships 3.12) or 22.04 (ships
3.10). Install it via the deadsnakes PPA:

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

Verify:
```bash
python3.11 --version
```
Should print `Python 3.11.x`.

---

## 2. System packages needed

```bash
sudo apt install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    git
```
- `libgl1`/`libglib2.0-0` — OpenCV needs these even in headless/server use
- `ffmpeg` — video decode fallback (same role it played on Windows)
- `build-essential` — some Python packages compile small C extensions on install

**Optional but recommended — real GStreamer support:** unlike Windows
(where `gi`/GStreamer Python bindings aren't readily available and the
system fell back to OpenCV+FFmpeg with 1-2s extra latency), Ubuntu can run
GStreamer natively for lower-latency RTSP decode:
```bash
sudo apt install -y \
    python3-gi \
    gir1.2-gstreamer-1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav
```
Not required — the FFmpeg fallback works fine — but worth doing if you
want to shave off that residual lag. `camera_reader.py` already tries
GStreamer first automatically; installing this just makes that path
actually available instead of always falling through to FFmpeg.

---

## 3. Project setup

Unzip the project wherever you like, e.g.:
```bash
mkdir -p ~/facetrack
cd ~/facetrack
unzip facetrack_project_hp.zip
cd facetrack_project_hp
```

Create the venv with **python3.11 specifically** (not plain `python3`,
which may point to a different version):
```bash
python3.11 -m venv venv
source venv/bin/activate
```
Note the `source ... activate` syntax — this replaces Windows'
`venv\Scripts\activate`. You'll need to run this in every new terminal
session, same as before.

Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Configuration — same as before, just check these got copied

- **`.env`** — your `GEMINI_API_KEY=AIza...` (this file isn't in the zip
  for security — you'll need to recreate it)
- **`config.py`** — your two HLC camera RTSP URLs
  (`rtsp://admin:Admin%40123@192.168.1.102:554/Streaming/Channels/101` and
  `.101`) should already be in the file as shipped. Same cameras, same
  network, should just work — verify with `ping` if not.

Create `.env`:
```bash
nano .env
```
Paste:
```
GEMINI_API_KEY=AIza...
```
Save (Ctrl+O, Enter, Ctrl+X in nano).

---

## 5. Test run before setting up auto-start

```bash
python main.py
```
Confirm: database ready, InsightFace loads, both HLC cameras connect,
dashboard loads at `http://localhost:8001`. Ctrl+C to stop once confirmed.

---

## 6. Always-on setup — systemd (replaces Task Scheduler + the .bat wrapper)

This is actually simpler than the Windows setup — systemd has built-in
auto-restart, no wrapper script/loop needed.

Create the service file:
```bash
sudo nano /etc/systemd/system/facetrack.service
```
Paste (**edit the `User`, `WorkingDirectory`, and paths to match your
actual username and install location**):
```ini
[Unit]
Description=FaceTrack Pro
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/facetrack/facetrack_project_hp
ExecStart=/home/YOUR_USERNAME/facetrack/facetrack_project_hp/venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=append:/home/YOUR_USERNAME/facetrack/facetrack_project_hp/logs/systemd.log
StandardError=append:/home/YOUR_USERNAME/facetrack/facetrack_project_hp/logs/systemd.log

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable facetrack.service
sudo systemctl start facetrack.service
```

Check it's running:
```bash
sudo systemctl status facetrack.service
```

View live logs:
```bash
journalctl -u facetrack.service -f
```

Stop / restart (e.g. after a code update):
```bash
sudo systemctl restart facetrack.service
sudo systemctl stop facetrack.service
```

**What `Restart=always` + `enable` gives you** — same coverage as the
Windows Task Scheduler setup:
- Crashes → systemd restarts it automatically after 10s
- Reboots → starts automatically on boot, no login needed
  (`multi-user.target` = starts at boot regardless of whether anyone logs
  into the desktop)

---

## 7. Daily database backup — systemd timer (replaces the Windows scheduled task)

Backup script:
```bash
nano ~/facetrack/facetrack_project_hp/backup_attendance.sh
```
```bash
#!/bin/bash
cd /home/YOUR_USERNAME/facetrack/facetrack_project_hp
mkdir -p backups logs
TODAY=$(date +%Y-%m-%d)
cp database/attendance.db "backups/attendance_${TODAY}.db"
echo "[$(date)] Backed up attendance.db to backups/attendance_${TODAY}.db" >> logs/backup.log
# Delete backups older than 30 days
find backups -name "attendance_*.db" -mtime +30 -delete
```
```bash
chmod +x ~/facetrack/facetrack_project_hp/backup_attendance.sh
```

Systemd service + timer:
```bash
sudo nano /etc/systemd/system/facetrack-backup.service
```
```ini
[Unit]
Description=FaceTrack Pro daily backup

[Service]
Type=oneshot
User=YOUR_USERNAME
ExecStart=/home/YOUR_USERNAME/facetrack/facetrack_project_hp/backup_attendance.sh
```

```bash
sudo nano /etc/systemd/system/facetrack-backup.timer
```
```ini
[Unit]
Description=Run FaceTrack backup daily at 2 AM

[Timer]
OnCalendar=*-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

Enable:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now facetrack-backup.timer
```

Verify it's scheduled:
```bash
systemctl list-timers facetrack-backup.timer
```

Run it manually right now to test (don't wait for 2 AM):
```bash
sudo systemctl start facetrack-backup.service
cat ~/facetrack/facetrack_project_hp/logs/backup.log
```

---

## 8. Log rotation — no change needed

`main.py` already configures loguru with daily rotation + 14-day
retention (`logs/facetrack_YYYY-MM-DD.log`) — this is pure Python,
platform-independent, works identically on Linux.

---

## Summary — what's different from Windows

| Thing | Windows | Ubuntu |
|---|---|---|
| Python version needed | 3.11 (via py launcher) | 3.11 (via deadsnakes PPA) |
| venv activate | `venv\Scripts\activate` | `source venv/bin/activate` |
| Auto-restart on crash | `run_facetrack.bat` loop + Task Scheduler | `systemctl` with `Restart=always` (built-in, no wrapper script) |
| Runs on boot, no login | Task Scheduler, `/ru SYSTEM` | systemd `enable`, `multi-user.target` (built-in) |
| Daily backup | Task Scheduler `/sc daily` | systemd timer |
| View logs | Redirect to file manually | `journalctl -u facetrack.service -f` (built-in) + loguru's own rotated files |
| GStreamer camera decode | Not available (falls back to FFmpeg) | Available if you install the packages in step 2 — lower latency |

Everything else — `config.py`, the camera RTSP URLs, `.env` format,
`enroll.py` commands, the dashboard, the attendance policy, the HLC
door-based detection — is identical. This was all pure Python from the
start; only the OS-level plumbing around it changes.
