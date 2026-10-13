"""
FaceTrack Pro — hr_routes.py
FastAPI router for HR admin endpoints. Mounted at /hr.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from hr_database import hr_db, LEAVE_LABELS

router = APIRouter(prefix="/hr", tags=["HR"])


# ── Request models ─────────────────────────────────────────────

class OverrideRequest(BaseModel):
    employee_id:   str
    work_date:     str
    new_status:    str
    reason:        str
    overridden_by: str = "admin"
    first_seen:    Optional[str] = None
    last_seen:     Optional[str] = None


class LeaveApplyRequest(BaseModel):
    employee_id:  str
    leave_type:   str
    from_date:    str
    to_date:      str
    days:         float
    reason:       str = ""
    document_ref: str = ""


class ActionRequest(BaseModel):
    approved_by: str = "admin"
    remarks:     str = ""


class RejectRequest(BaseModel):
    rejected_by: str = "admin"
    reason:      str


class ODApplyRequest(BaseModel):
    employee_id:     str
    od_date:         str
    purpose:         str
    destination:     str = ""
    session:         str = "FULL"
    expected_return: str = ""


# ── Attendance override ────────────────────────────────────────

@router.post("/override")
async def manual_override(req: OverrideRequest):
    result = hr_db.manual_override(
        req.employee_id, req.work_date, req.new_status,
        req.reason, req.overridden_by, req.first_seen, req.last_seen,
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/overrides")
async def get_overrides(employee_id: str = None, work_date: str = None):
    return hr_db.get_override_history(employee_id, work_date)


# ── Leave ──────────────────────────────────────────────────────

@router.get("/leaves")
async def leaves_flat():
    """Flat list of pending leave applications, for dashboard tables."""
    return hr_db.get_pending_applications()["leave"]


@router.get("/balance/{employee_id}")
async def balance_flat(employee_id: str, year: int = None):
    """Flattened leave balance shape for the dashboard balance widget."""
    b = hr_db.get_leave_balance(employee_id, year=year)
    return {
        "cl_remaining": b.get("CL", {}).get("balance", 0),
        "ml_remaining": b.get("ML", {}).get("balance", 0),
        "el_remaining": b.get("EL", {}).get("balance", 0),
        "perm_this_month": 0,  # not tracked in this schema yet — see HP_DEPLOYMENT_CHANGES.md
    }


@router.get("/leave/types")
async def leave_types():
    return [{"code": k, "label": v} for k, v in LEAVE_LABELS.items()]


@router.post("/leave/apply")
async def apply_leave(req: LeaveApplyRequest):
    if req.leave_type not in LEAVE_LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid leave type. Valid types: {list(LEAVE_LABELS.keys())}",
        )
    result = hr_db.apply_leave(
        req.employee_id, req.leave_type, req.from_date,
        req.to_date, req.days, req.reason, req.document_ref,
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/leave/approve/{application_id}")
async def approve_leave(application_id: int, req: ActionRequest):
    result = hr_db.approve_leave(application_id, req.approved_by, req.remarks)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/leave/reject/{application_id}")
async def reject_leave(application_id: int, req: RejectRequest):
    result = hr_db.reject_leave(application_id, req.rejected_by, req.reason)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ── On-Duty ────────────────────────────────────────────────────

@router.post("/od/apply")
async def apply_od(req: ODApplyRequest):
    if req.session not in ("FULL", "AM", "PM"):
        raise HTTPException(status_code=400, detail="session must be FULL, AM, or PM")
    result = hr_db.apply_od(
        req.employee_id, req.od_date, req.purpose,
        req.destination, req.session, req.expected_return,
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/od/approve/{application_id}")
async def approve_od(application_id: int, req: ActionRequest):
    result = hr_db.approve_od(application_id, req.approved_by, req.remarks)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ── Queries ────────────────────────────────────────────────────

@router.get("/pending")
async def pending_applications():
    return hr_db.get_pending_applications()


@router.get("/balances")
async def all_balances(year: int = None):
    return hr_db.get_all_leave_balances(year)


@router.get("/balances/{employee_id}")
async def faculty_balance(employee_id: str, year: int = None):
    return hr_db.get_leave_balance(employee_id, year=year)


@router.post("/balances/init/{employee_id}")
async def init_balances(employee_id: str, year: int = None):
    hr_db.initialize_leave_balances(employee_id, year)
    return {"success": True}


@router.get("/faculty/{employee_id}/history")
async def faculty_history(employee_id: str, year: int = None):
    return hr_db.get_faculty_leave_history(employee_id, year)
