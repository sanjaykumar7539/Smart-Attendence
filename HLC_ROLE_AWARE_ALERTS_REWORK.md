# HLC Role-Aware Overstay Alerts — Rework (Phase 1)

Scope, as agreed: **HLC members vs non-HLC members only.** Full student
and all-paid-employee attendance is Phase 2 — not touched here.

---

## What changed and why

The HLC pipeline already had the right skeleton (team vs visitor split,
15-min-style overstay tracking, alert engine) but three things were
broken or missing before this rework:

1. **`emp_info` was never actually populated.** `main.py` built the HLC
   detection dict without it, so `hlc_processor_v4.py` always fell back to
   `dept=""` and **hardcoded `person_type="FACULTY"`** for every non-HLC
   person — an Admin Executive and an Associate Professor looked
   identical in every alert and report.
2. **Two disconnected "is this person HLC" records** — `faculty.is_hlc`
   (master DB, set by `enroll.py --hlc`) vs `hlc_employees` in the
   separate `hlc_team.db` (also settable independently via the
   `/api/hlc/team/add` dashboard endpoint). These could drift apart.
3. **Alert threshold was 5 minutes**, not the 15 you specified.
   (`config.py` → `HLCConfig.alert_threshold_mins`)
4. **No actual push notification** — alerts only reached the dashboard's
   in-page toast via WebSocket. Nothing fired a real OS-level
   notification, and there was no path to email/WhatsApp at all.

## What this rework does

- **Real role data flows through.** `main.py` now looks up
  `db.get_faculty(employee_id)` — `department`, `designation`,
  `staff_category` (Teaching/Non-Teaching), `is_hlc` — and passes it as
  `emp_info` on every HLC detection.
- **`faculty.is_hlc` is now the source of truth** for HLC membership,
  with `hlc_team.db` kept only as a fallback for any legacy entries not
  yet migrated (no forced data migration needed — nothing breaks on your
  existing `hlc_team.db`).
- **Non-HLC sessions now carry real role data.** `person_type` on a
  visitor session is the actual `staff_category` ("Teaching" /
  "Non-Teaching"), and a new `designation` column
  (e.g. "Associate Professor", "Office Executive") is stored and shown
  in every alert, the live alert feed, and the historical alerts table.
- **Threshold corrected to 15 minutes.**
- **Browser push notifications are live now.** The dashboard requests
  Notification permission on load; any HLC overstay alert fires a real
  OS-level notification (not just the toast) whenever the dashboard tab
  is open. Click it and it jumps you to the HLC tab.
- **Email + WhatsApp are wired but off.** `config.py` has a new
  `NotifyConfig` block (`config.notify.email`, `config.notify.whatsapp`)
  with `enabled=False` and empty credential fields. The `visit_alerts`
  table already has `notified_email` / `notified_whatsapp` columns and
  the alert engine already calls stub `_send_email()` / `_send_whatsapp()`
  hooks — when you're ready for Phase 2 notification channels, it's
  just: fill in SMTP/WhatsApp creds in `config.py`, flip `enabled=True`,
  and implement the two stub methods in `hlc_alert_engine.py`. No schema
  or wiring changes needed at that point.
- **Bonus fix, same code path:** the alerts-history table on the
  dashboard was silently ignoring the date filter (`work_date` param
  sent by the frontend, but the backend route expects `date`) — fixed.

## Files changed

- `config.py` — threshold 5→15 min, new `NotifyConfig` block
- `main.py` — builds and passes `emp_info` on every HLC detection
- `hlc_processor_v4.py` — uses real role data, `faculty.is_hlc` as
  primary membership check
- `hlc_visitor_db_v4.py` — new `designation` column (auto-migrates
  existing `hlc_visitors.db`), `notified_email` column, join for
  designation in alert history
- `hlc_alert_engine.py` — richer alert message/payload, `push` flag,
  email/WhatsApp stub hooks (off by default)
- `static/dashboard.html` — Notification permission + real push on
  alert, role shown in live alert card and history table, date-filter
  fix

## Deploying on the HP tower

Your live DBs (`database/attendance.db`, `hlc_team.db`,
`hlc_visitors.db`) are **not touched or replaced** — the schema
migrations run automatically on next startup and only *add* columns.

```bash
# on the HP tower
cd /home/iotadmin/facetrack/facetrack_project_complete

# stop the service first
sudo systemctl stop facetrack.service

# back up DBs just in case (cheap insurance)
mkdir -p backups
cp database/attendance.db hlc_team.db hlc_visitors.db backups/ 2>/dev/null

# copy in the changed files (from wherever you transfer the patch to)
# config.py, main.py, hlc_processor_v4.py, hlc_visitor_db_v4.py,
# hlc_alert_engine.py, static/dashboard.html

# quick manual smoke test before trusting systemd with it
source venv/bin/activate
python main.py
# watch the startup log for:
#   "Migrated visitor_sessions: added designation column"
#   "Migrated visit_alerts: added notified_email column"
# open the dashboard, allow browser notifications when prompted,
# check HLC tab loads with no errors. Ctrl+C once confirmed.

sudo systemctl start facetrack.service
sudo systemctl status facetrack.service
```

## Making someone an HLC member going forward

Use `enroll.py --hlc` as before — it now writes `is_hlc=1` into the
master `faculty` table (the thing that actually matters now) in
addition to `hlc_team.db`:

```bash
python enroll.py --id PRIN001 --name "Dr. ..." --dept "Administration" \
  --desig "Principal" --category Non-Teaching --hlc --images img1.jpg
```

If you add/remove HLC members via the dashboard's team-management
endpoints instead, note that today it still only touches `hlc_team.db`
— it'll work (fallback check still applies) but won't set the
`faculty.is_hlc` flag. Not a blocker, just something to be aware of
until that endpoint is also pointed at the master table.

## Next (Phase 2, once this is confirmed stable)

- Extend `staff_category`/`designation` model to full campus attendance
  (students + all paid employees), not just HLC
- Turn on email and/or WhatsApp channels (`config.notify.*`)
- Point the dashboard's "Add HLC member" endpoint at `faculty.is_hlc`
  directly so there's truly one source of truth
