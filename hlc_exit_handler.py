"""hlc_exit_handler.py — Direction confirmation buffer + session timeout sweep"""
import threading, logging, time
from datetime import datetime, date, timedelta
from typing import Dict, Optional
logger = logging.getLogger(__name__)

class DirectionBuffer:
    """N consecutive same-direction frames before accepting."""
    def __init__(self, confirm_frames=3):
        self.confirm_frames=confirm_frames
        self._buf: Dict[str,list]={}

    def push(self, track_id: str, direction: str) -> Optional[str]:
        if track_id not in self._buf: self._buf[track_id]=[]
        self._buf[track_id].append(direction)
        self._buf[track_id]=self._buf[track_id][-self.confirm_frames:]
        buf=self._buf[track_id]
        if len(buf)==self.confirm_frames and len(set(buf))==1:
            confirmed=buf[0]; self._buf[track_id]=[]
            return confirmed
        return None

    def clear(self, track_id: str): self._buf.pop(track_id,None)


class SessionTimeoutSweep:
    """Background thread — auto-closes sessions open > timeout_hours."""
    def __init__(self, visitor_db, timeout_hours=4.0, check_interval_mins=30):
        self.vdb=visitor_db; self.timeout=timeout_hours
        self.interval=check_interval_mins*60
        self._running=False; self._thread=None

    def start(self):
        self._running=True
        self._thread=threading.Thread(target=self._loop,daemon=True,name="SessionTimeoutSweep")
        self._thread.start()
        logger.info(f"Session timeout sweep started | timeout={self.timeout}h")

    def stop(self): self._running=False

    def _loop(self):
        while self._running:
            try: self._sweep()
            except Exception as e: logger.error(f"Sweep error: {e}")
            time.sleep(self.interval)

    def _sweep(self):
        today=date.today().isoformat()
        open_sessions=self.vdb.get_all_open_sessions(today)
        cutoff_minutes=self.timeout*60
        closed=0
        for sess in open_sessions:
            elapsed=sess.get("minutes_so_far",0) or 0
            if elapsed>=cutoff_minutes:
                try:
                    from datetime import datetime as dt
                    entry_dt=dt.strptime(f"{sess['visit_date']} {sess['entry_time']}","%Y-%m-%d %H:%M:%S")
                    exit_dt=entry_dt+timedelta(hours=self.timeout)
                    exit_time=exit_dt.strftime("%H:%M:%S")
                except Exception:
                    exit_time=datetime.now().strftime("%H:%M:%S")
                self.vdb.close_session_with_method(sess["id"],exit_time,"TIMEOUT")
                closed+=1
                logger.warning(f"TIMEOUT CLOSE | session={sess['id']} | {sess['visitor_id']} | elapsed={elapsed:.0f}m")
        if closed: logger.info(f"Sweep: auto-closed {closed} stale session(s)")
