"""hlc_export.py — CSV/JSON export helpers for HLC data"""
import io, csv, json
from typing import List, Dict, Any

def to_csv(rows: List[Dict], columns: List[str]=None) -> bytes:
    if not rows: return b""
    cols=columns or list(rows[0].keys())
    buf=io.StringIO()
    w=csv.DictWriter(buf,fieldnames=cols,extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")

def to_json(data: Any) -> bytes:
    return json.dumps(data,ensure_ascii=False,indent=2,default=str).encode("utf-8")

def visitor_csv(rows: List[Dict]) -> bytes:
    cols=["id","visitor_id","name","dept","person_type","visit_date",
          "entry_time","exit_time","duration_minutes","visit_number",
          "flagged","over_15min","purpose","door_id","exit_door","exit_method","camera_id"]
    return to_csv(rows,cols)

def team_attendance_csv(rows: List[Dict]) -> bytes:
    cols=["employee_id","name","dept","hlc_role","work_date",
          "first_seen","last_seen","morning_status","evening_status","final_status","notes"]
    return to_csv(rows,cols)
