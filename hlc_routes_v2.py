"""
hlc_routes_v2.py — FastAPI router for all HLC endpoints.
Mount in main.py: app.include_router(hlc_router)
"""
import os, json, logging
from datetime import date
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)
hlc_router = APIRouter()

_tdb = None   # HLCTeamDB
_vdb = None   # HLCVisitorDBV4
_eng = None   # HLCAlertEngine

def init_hlc(team_db, visitor_db, alert_engine):
    global _tdb, _vdb, _eng
    _tdb = team_db
    _vdb = visitor_db
    _eng = alert_engine

def _today(): return date.today().isoformat()
def _tdb_req():
    if not _tdb: raise HTTPException(500, "HLC team DB not initialised")
    return _tdb
def _vdb_req():
    if not _vdb: raise HTTPException(500, "HLC visitor DB not initialised")
    return _vdb

# ── Pydantic models ───────────────────────────────────────────────────────────
class TeamAddReq(BaseModel):
    employee_id: str; name: str
    dept: Optional[str]=None; role: Optional[str]=None
    phone: Optional[str]=None; email: Optional[str]=None
    is_admin: bool=False

class OverrideReq(BaseModel):
    employee_id: str; work_date: str
    override_status: str; reason: str=""

class PurposeReq(BaseModel):
    purpose: str

class ResolveReq(BaseModel):
    resolved_by: str; note: Optional[str]=""

# ── Status ────────────────────────────────────────────────────────────────────
@hlc_router.get("/api/hlc/status")
def hlc_status():
    return {"hlc_enabled": _tdb is not None, "alert_engine": _eng is not None}

# ── Live presence ─────────────────────────────────────────────────────────────
@hlc_router.get("/api/hlc/stats")
def hlc_stats(work_date: str=Query(default=None)):
    d=work_date or _today()
    t=_tdb_req().get_stats(d); v=_vdb_req().get_stats(d)
    open_alerts=len(_vdb_req().get_open_alerts(d))
    return {**t, **v, "open_alerts": open_alerts, "date": d}

@hlc_router.get("/api/hlc/presence")
def hlc_presence():
    today=_today()
    team=_tdb_req().get_live_present()
    inside=_vdb_req().get_all_open_sessions(today)
    open_a=_vdb_req().get_open_alerts(today)
    return {"date":today,"hlc_team_present":team,"hlc_team_count":len(team),
            "visitors_inside":inside,"visitors_count":len(inside),
            "open_alerts":open_a,"alert_count":len(open_a)}

# ── HLC Team ──────────────────────────────────────────────────────────────────
@hlc_router.get("/api/hlc/team/members")
def list_team(): return _tdb_req().get_all_employees()

@hlc_router.post("/api/hlc/team/add")
def add_team(req: TeamAddReq):
    _tdb_req().add_employee(req.employee_id,req.name,req.dept,req.role,req.phone,req.email,req.is_admin)
    return {"message": f"{req.employee_id} added to HLC team"}

@hlc_router.post("/api/hlc/team/remove")
def remove_team(req: TeamAddReq):
    _tdb_req().remove_employee(req.employee_id)
    return {"message": f"{req.employee_id} removed from HLC team"}

@hlc_router.get("/api/hlc/team/attendance")
def team_attendance(work_date: str=Query(default=None)):
    return _tdb_req().get_attendance(work_date or _today())

@hlc_router.post("/api/hlc/team/attendance/override")
def team_override(req: OverrideReq):
    valid={"PRESENT","ABSENT","PERMISSION","HALF_DAY","CL","ML","EL","OD"}
    if req.override_status not in valid:
        raise HTTPException(400, f"Invalid status: {req.override_status}")
    _tdb_req().override_attendance(req.employee_id,req.work_date,req.override_status,req.reason)
    return {"message": f"Override: {req.employee_id} → {req.override_status}"}

# ── Visitors ──────────────────────────────────────────────────────────────────
@hlc_router.get("/api/hlc/visitors")
def visitor_log(date: str=Query(default=None), from_date: str=Query(default=None),
                to_date: str=Query(default=None), visitor_id: str=Query(default=None)):
    vdb=_vdb_req()
    return vdb.get_sessions(visit_date=date or (_today() if not from_date else None),
                            visitor_id=visitor_id, from_date=from_date, to_date=to_date)

@hlc_router.get("/api/hlc/visitors/live")
def visitors_live(): return _vdb_req().get_all_open_sessions(_today())

@hlc_router.get("/api/hlc/visitors/summary")
def visitor_summary(visit_date: str=Query(default=None)):
    return _vdb_req().get_daily_summary(visit_date or _today())

@hlc_router.get("/api/hlc/visitor/{visitor_id}")
def visitor_history(visitor_id: str, from_date: str=Query(default=None), to_date: str=Query(default=None)):
    rows=_vdb_req().get_sessions(visitor_id=visitor_id,from_date=from_date,to_date=to_date)
    total_dur=sum(r["duration_minutes"] for r in rows if r.get("duration_minutes"))
    return {"visitor_id":visitor_id,"name":rows[0].get("name") if rows else None,
            "total_visits":len(rows),"total_minutes":round(total_dur,1),"visits":rows}

@hlc_router.post("/api/hlc/visitor/{session_id}/purpose")
def set_purpose(session_id: int, req: PurposeReq):
    _vdb_req().update_purpose(session_id,req.purpose)
    return {"message":"Purpose updated"}

# ── Alerts ────────────────────────────────────────────────────────────────────
@hlc_router.get("/api/hlc/alerts")
def all_alerts(date: str=Query(default=None), from_date: str=Query(default=None), to_date: str=Query(default=None)):
    return _vdb_req().get_all_alerts(visit_date=date,from_date=from_date,to_date=to_date)

@hlc_router.get("/api/hlc/alerts/open")
def open_alerts(visit_date: str=Query(default=None)):
    return _vdb_req().get_open_alerts(visit_date or _today())

@hlc_router.post("/api/hlc/alert/{alert_id}/resolve")
def resolve_alert(alert_id: int, req: ResolveReq):
    _vdb_req().resolve_alert(alert_id,req.resolved_by,req.note)
    return {"message":f"Alert {alert_id} resolved"}

@hlc_router.post("/api/hlc/alert/force_check")
def force_check():
    if _eng: _eng.force_check(); return {"message":"Alert check triggered"}
    return {"message":"Alert engine not running"}

# ── EOD Reports ───────────────────────────────────────────────────────────────
def _eod(report_date=None):
    from hlc_eod_report import assemble_eod_data
    d=report_date or _today()
    return assemble_eod_data(_tdb_req(),_vdb_req(),d), d

@hlc_router.get("/api/hlc/eod/json")
def eod_json(report_date: str=Query(default=None)):
    from hlc_eod_report import generate_eod_json
    data,d=_eod(report_date)
    return Response(content=generate_eod_json(data),media_type="application/json",
                    headers={"Content-Disposition":f'attachment; filename="hlc_eod_{d}.json"'})

@hlc_router.get("/api/hlc/eod/pdf")
def eod_pdf(report_date: str=Query(default=None)):
    from hlc_eod_report import generate_eod_pdf
    data,d=_eod(report_date)
    return Response(content=generate_eod_pdf(data),media_type="application/pdf",
                    headers={"Content-Disposition":f'attachment; filename="hlc_eod_{d}.pdf"'})

@hlc_router.get("/api/hlc/eod/xlsx")
def eod_xlsx(report_date: str=Query(default=None)):
    from hlc_eod_report import generate_eod_xlsx
    data,d=_eod(report_date)
    return Response(content=generate_eod_xlsx(data),
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition":f'attachment; filename="hlc_eod_{d}.xlsx"'})

@hlc_router.get("/api/hlc/eod/csv")
def eod_csv(report_date: str=Query(default=None)):
    from hlc_eod_report import generate_eod_csv_zip
    data,d=_eod(report_date)
    return Response(content=generate_eod_csv_zip(data),media_type="application/zip",
                    headers={"Content-Disposition":f'attachment; filename="hlc_eod_{d}.zip"'})

# ── Per-dataset exports ───────────────────────────────────────────────────────
@hlc_router.get("/api/hlc/export/visitors/csv")
def export_vis_csv(date: str=Query(default=None), from_date: str=Query(default=None), to_date: str=Query(default=None)):
    from hlc_export import visitor_csv
    f,t=(from_date or date or _today()),(to_date or date or _today())
    return Response(content=visitor_csv(_vdb_req().get_sessions(from_date=f,to_date=t)),
                    media_type="text/csv",headers={"Content-Disposition":f'attachment; filename="hlc_visitors_{f}.csv"'})

@hlc_router.get("/api/hlc/export/team/csv")
def export_team_csv(work_date: str=Query(default=None), from_date: str=Query(default=None), to_date: str=Query(default=None)):
    from hlc_export import team_attendance_csv
    rows=(_tdb_req().get_attendance_range(from_date or _today(), to_date or _today())
          if from_date else _tdb_req().get_attendance(work_date or _today()))
    d=work_date or from_date or _today()
    return Response(content=team_attendance_csv(rows),media_type="text/csv",
                    headers={"Content-Disposition":f'attachment; filename="hlc_team_{d}.csv"'})
