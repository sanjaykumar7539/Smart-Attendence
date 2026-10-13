"""hlc_team_db.py — HLC team attendance (hlc_team.db)"""
import sqlite3, threading, logging
from datetime import date
from typing import Optional, List, Dict
from contextlib import contextmanager
logger = logging.getLogger(__name__)

class HLCTeamDB:
    def __init__(self, db_path="hlc_team.db"):
        self.db_path = db_path
        self._local  = threading.local()
        self._init()

    def _conn(self):
        if not hasattr(self._local,"conn") or self._local.conn is None:
            c = sqlite3.connect(self.db_path, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            self._local.conn = c
        return self._local.conn

    @contextmanager
    def _cur(self):
        conn=self._conn(); cur=conn.cursor()
        try: yield cur; conn.commit()
        except: conn.rollback(); raise
        finally: cur.close()

    def _init(self):
        with self._cur() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS hlc_employees(
                employee_id TEXT PRIMARY KEY, name TEXT NOT NULL,
                dept TEXT, hlc_role TEXT, phone TEXT, email TEXT,
                is_admin INTEGER DEFAULT 0, is_active INTEGER DEFAULT 1)""")
            c.execute("""CREATE TABLE IF NOT EXISTS hlc_attendance(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL, work_date TEXT NOT NULL,
                first_seen TEXT, last_seen TEXT,
                morning_status TEXT DEFAULT 'ABSENT',
                evening_status TEXT, final_status TEXT DEFAULT 'ABSENT',
                notes TEXT, override_status TEXT, override_reason TEXT,
                created_at TEXT DEFAULT(datetime('now','localtime')),
                updated_at TEXT DEFAULT(datetime('now','localtime')),
                UNIQUE(employee_id,work_date))""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_hta_date ON hlc_attendance(work_date)")
        logger.info(f"HLCTeamDB ready: {self.db_path}")

    def add_employee(self, employee_id, name, dept=None, role=None,
                     phone=None, email=None, is_admin=False):
        with self._cur() as c:
            c.execute("""INSERT INTO hlc_employees
                (employee_id,name,dept,hlc_role,phone,email,is_admin)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(employee_id) DO UPDATE SET
                name=excluded.name, dept=excluded.dept,
                hlc_role=excluded.hlc_role, is_admin=excluded.is_admin""",
                (employee_id,name,dept,role,phone,email,int(is_admin)))

    def remove_employee(self, employee_id):
        with self._cur() as c:
            c.execute("UPDATE hlc_employees SET is_active=0 WHERE employee_id=?",(employee_id,))

    def is_hlc_employee(self, employee_id) -> bool:
        with self._cur() as c:
            c.execute("SELECT 1 FROM hlc_employees WHERE employee_id=? AND is_active=1",(employee_id,))
            return c.fetchone() is not None

    def get_all_employees(self) -> List[Dict]:
        with self._cur() as c:
            c.execute("SELECT * FROM hlc_employees WHERE is_active=1 ORDER BY name")
            return [dict(r) for r in c.fetchall()]

    def get_employee(self, employee_id) -> Optional[Dict]:
        with self._cur() as c:
            c.execute("SELECT * FROM hlc_employees WHERE employee_id=?",(employee_id,))
            r=c.fetchone(); return dict(r) if r else None

    def get_admins(self) -> List[Dict]:
        with self._cur() as c:
            c.execute("SELECT * FROM hlc_employees WHERE is_admin=1 AND is_active=1")
            return [dict(r) for r in c.fetchall()]

    def record_entry(self, employee_id, work_date, time_str) -> Dict:
        with self._cur() as c:
            c.execute("SELECT id FROM hlc_attendance WHERE employee_id=? AND work_date=?",(employee_id,work_date))
            if c.fetchone():
                c.execute("UPDATE hlc_attendance SET last_seen=?,updated_at=datetime('now','localtime') WHERE employee_id=? AND work_date=?",(time_str,employee_id,work_date))
                return {"action":"updated"}
            status=self._policy(time_str)
            emp=self.get_employee(employee_id)
            name=emp["name"] if emp else employee_id
            c.execute("INSERT INTO hlc_attendance(employee_id,work_date,first_seen,last_seen,morning_status,final_status) VALUES(?,?,?,?,?,?)",(employee_id,work_date,time_str,time_str,status,status))
            return {"action":"created","status":status}

    def record_exit(self, employee_id, work_date, time_str):
        with self._cur() as c:
            c.execute("UPDATE hlc_attendance SET last_seen=?,updated_at=datetime('now','localtime') WHERE employee_id=? AND work_date=?",(time_str,employee_id,work_date))

    def _policy(self, t) -> str:
        try:
            h,m,*_=map(int,t.split(":"))
            mins=h*60+m
            if mins<=9*60+5:  return "PRESENT"
            if mins<=9*60+55: return "PERMISSION"
            return "ABSENT"
        except: return "PRESENT"

    def override_attendance(self, employee_id, work_date, status, reason):
        with self._cur() as c:
            c.execute("""INSERT INTO hlc_attendance(employee_id,work_date,final_status,override_status,override_reason)
                VALUES(?,?,?,?,?) ON CONFLICT(employee_id,work_date) DO UPDATE SET
                final_status=excluded.final_status,override_status=excluded.override_status,
                override_reason=excluded.override_reason,updated_at=datetime('now','localtime')""",
                (employee_id,work_date,status,status,reason))

    def get_attendance(self, work_date) -> List[Dict]:
        with self._cur() as c:
            c.execute("""SELECT a.*,e.dept,e.hlc_role FROM hlc_attendance a
                LEFT JOIN hlc_employees e ON a.employee_id=e.employee_id
                WHERE a.work_date=? ORDER BY a.first_seen""",(work_date,))
            return [dict(r) for r in c.fetchall()]

    def get_attendance_range(self, from_date, to_date) -> List[Dict]:
        with self._cur() as c:
            c.execute("""SELECT a.*,e.name,e.dept,e.hlc_role FROM hlc_attendance a
                JOIN hlc_employees e ON a.employee_id=e.employee_id
                WHERE a.work_date BETWEEN ? AND ? ORDER BY a.work_date,e.name""",(from_date,to_date))
            return [dict(r) for r in c.fetchall()]

    def get_live_present(self) -> List[Dict]:
        today=date.today().isoformat()
        with self._cur() as c:
            c.execute("""SELECT a.employee_id,e.name,e.hlc_role,a.first_seen
                FROM hlc_attendance a JOIN hlc_employees e ON a.employee_id=e.employee_id
                WHERE a.work_date=? AND a.final_status IN('PRESENT','PERMISSION','HALF_DAY')
                ORDER BY a.first_seen""",(today,))
            return [dict(r) for r in c.fetchall()]

    def get_stats(self, work_date) -> Dict:
        with self._cur() as c:
            c.execute("""SELECT COUNT(*) as total_team,
                SUM(CASE WHEN final_status='PRESENT' THEN 1 ELSE 0 END) as present,
                SUM(CASE WHEN final_status='ABSENT'  THEN 1 ELSE 0 END) as absent,
                SUM(CASE WHEN final_status='PERMISSION' THEN 1 ELSE 0 END) as permission,
                SUM(CASE WHEN final_status='HALF_DAY' THEN 1 ELSE 0 END) as half_day
                FROM hlc_attendance WHERE work_date=?""",(work_date,))
            stats=dict(c.fetchone() or {})
            c.execute("SELECT COUNT(*) as enrolled FROM hlc_employees WHERE is_active=1")
            stats["enrolled"]=c.fetchone()["enrolled"]
            return stats
