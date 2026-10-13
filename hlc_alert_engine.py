"""hlc_alert_engine.py — Background 15-min overstay alert engine"""
import threading, logging, time
from datetime import datetime, date
from typing import Callable, Optional, Dict, Any
from config import config
logger = logging.getLogger(__name__)

class HLCAlertEngine:
    def __init__(self, visitor_db, team_db, check_interval_secs=60, threshold_minutes=15.0):
        self.vdb=visitor_db; self.tdb=team_db
        self.interval=check_interval_secs; self.threshold=threshold_minutes
        self._running=False; self._thread=None; self._ws_broadcast=None

    def start(self):
        if self._running: return
        self._running=True
        self._thread=threading.Thread(target=self._loop,daemon=True,name="HLCAlertEngine")
        self._thread.start()
        logger.info(f"HLC Alert Engine started | threshold={self.threshold}m | interval={self.interval}s")

    def stop(self): self._running=False

    def set_ws_broadcast(self, fn: Callable): self._ws_broadcast=fn

    def _loop(self):
        while self._running:
            try: self._check()
            except Exception as e: logger.error(f"Alert engine error: {e}",exc_info=True)
            time.sleep(self.interval)

    def _check(self):
        sessions=self.vdb.sessions_needing_alert(self.threshold)
        if not sessions: return
        logger.info(f"Alert check: {len(sessions)} session(s) over {self.threshold}m")
        for sess in sessions:
            alert_id=self.vdb.create_alert(
                session_id=sess["id"], visitor_id=sess["visitor_id"],
                visitor_name=sess["name"] or "Unknown", dept=sess["dept"] or "",
                visit_date=sess["visit_date"], entry_time=sess["entry_time"],
                duration_at_trigger=sess.get("minutes_elapsed",self.threshold))
            if alert_id==-1: continue
            self._dispatch(alert_id,sess)

    def _dispatch(self, alert_id, session):
        name  = session["name"] or "Unknown"
        dept  = session["dept"] or "—"
        desig = session.get("designation") or ""
        cat   = session.get("person_type") or ""       # "Teaching" | "Non-Teaching"
        role_label = " / ".join(x for x in (desig, cat) if x) or "—"
        entry=session["entry_time"]; duration=session.get("minutes_elapsed",self.threshold)
        visit_num=session.get("visit_number",1)
        message=(f"⚑ HLC ALERT — {name} ({session['visitor_id']})\n"
                 f"Role: {role_label} | Dept: {dept}\n"
                 f"Entry: {entry} | Inside: {duration:.0f} mins\n"
                 f"Visit #{visit_num} today")
        payload={"type":"hlc_alert","alert_id":alert_id,"visitor_id":session["visitor_id"],
                 "name":name,"dept":dept,"designation":desig,"staff_category":cat,
                 "entry_time":entry,
                 "duration_min":duration,"visit_number":visit_num,
                 "message":message,"timestamp":datetime.now().strftime("%H:%M:%S"),
                 # Tells the dashboard JS to also fire a real OS-level
                 # Notification (not just the in-page toast). Server-side
                 # "sent" flag below is what matters for the alert_log —
                 # this is just a hint for the browser.
                 "push": True}
        if self._ws_broadcast:
            import asyncio
            try:
                loop=asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._ws_broadcast({"type":"hlc_event","payload":payload}))
                else:
                    asyncio.run(self._ws_broadcast({"type":"hlc_event","payload":payload}))
            except Exception as e: logger.error(f"WS broadcast failed: {e}")
        self.vdb.log_alert_sent(alert_id,"dashboard","all_clients",message,"sent")
        logger.warning(f"Alert dispatched | alert_id={alert_id} | {name} ({role_label}) | {duration:.0f}m")

        # ── Future channels — off until credentials are configured ──────────
        # Both are already wired to alert_log/visit_alerts columns
        # (notified_email / notified_whatsapp), so turning these on later is
        # just: fill in config.notify.email / config.notify.whatsapp and set
        # enabled=True — no code or schema changes needed at that point.
        if config.notify.email.enabled:
            self._send_email(alert_id, message, session)
        if config.notify.whatsapp.enabled:
            self._send_whatsapp(alert_id, message, session)

    def _send_email(self, alert_id, message, session):
        """Stub — implement with smtplib once config.notify.email creds are set."""
        try:
            logger.info(f"[email stub] would send alert {alert_id} to {config.notify.email.to_addrs}")
            self.vdb.log_alert_sent(alert_id,"email",",".join(config.notify.email.to_addrs),message,"stub_not_sent")
        except Exception as e:
            logger.error(f"Email alert failed: {e}")

    def _send_whatsapp(self, alert_id, message, session):
        """Stub — implement with Twilio/WhatsApp Business API once config.notify.whatsapp creds are set."""
        try:
            logger.info(f"[whatsapp stub] would send alert {alert_id} to {config.notify.whatsapp.to_numbers}")
            self.vdb.log_alert_sent(alert_id,"whatsapp",",".join(config.notify.whatsapp.to_numbers),message,"stub_not_sent")
        except Exception as e:
            logger.error(f"WhatsApp alert failed: {e}")

    def force_check(self): self._check()
