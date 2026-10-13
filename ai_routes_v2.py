"""
ai_routes_v2.py
────────────────
FastAPI AI router — dual-mode API key support.

Server mode  (production):
  GEMINI_API_KEY set in .env → all requests use it automatically.
  No key needed in request headers.

Browser demo mode:
  Client sends header: X-Demo-Key: AIza...
  Server forwards it to Gemini.
  Useful for: demos without a server key, presentations, client-side.

Endpoints (mounted at /api/ai in main.py):
  GET  /status
  POST /chat          — #1 Admin Chatbot, single-turn {message,date} -> {answer}
  GET  /suggestions   — suggested question chips
  POST /dataset       — #2 Dataset Analysis (SSE)
  POST /analytics     — #3 Attendance Analytics (SSE)
  POST /report        — #4 Auto Report (SSE)
  POST /debug         — #5 Code Debug, all 4 types (SSE)
"""

import os
import logging
from datetime import date
from typing import Optional, List

from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from ai_engine_v2 import AIEngine

logger    = logging.getLogger(__name__)
ai_router = APIRouter()
_engine: Optional[AIEngine] = None


def init_ai(db_path:        str = "attendance.db",
             hlc_team_db:    str = "hlc_team.db",
             hlc_visitor_db: str = "hlc_visitors.db",
             api_key:        str = None):
    global _engine
    key = api_key or os.getenv("GEMINI_API_KEY", "")
    _engine = AIEngine(db_path=db_path, hlc_team_db=hlc_team_db,
                        hlc_visitor_db=hlc_visitor_db, api_key=key)
    logger.info(f"AI Engine ready | server key: {'✓' if key else '✗ (demo mode only)'}")


def eng() -> AIEngine:
    if _engine is None:
        raise RuntimeError("Call init_ai() in main.py first")
    return _engine


def sse(generator):
    """Wrap async generator as SSE StreamingResponse."""
    async def stream():
        async for chunk in generator:
            safe = chunk.replace("\n", "\\n")
            yield f"data: {safe}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache",
                                       "X-Accel-Buffering": "no",
                                       "Access-Control-Allow-Origin": "*"})


# ── Models ────────────────────────────────────────────────────────────────────

class ChatReq(BaseModel):
    message: str
    date:    Optional[str] = None

class AnalyticsReq(BaseModel):
    question:  str
    work_date: Optional[str] = None

class ReportReq(BaseModel):
    from_date:   str
    to_date:     str
    report_type: str = "monthly"

class DebugReq(BaseModel):
    error_msg:    str
    traceback:    str = ""
    code_snippet: str = ""
    debug_type:   str = "general"  # facetrack | sqlite | rtsp | general

class DatasetReq(BaseModel):
    days:  int = 30
    focus: str = ""


def local_chat_answer(message: str, work_date: str = None) -> str:
    """Answer common attendance questions without requiring a Gemini key."""
    day = work_date or date.today().isoformat()
    text = message.lower()
    enrolled = eng()._scalar(eng().db_path, "SELECT COUNT(*) FROM faculty") or 0
    rows = eng()._query(eng().db_path, """
        SELECT e.name, a.final_status, a.first_seen
        FROM attendance a LEFT JOIN faculty e ON e.employee_id=a.employee_id
        WHERE a.work_date=? ORDER BY a.first_seen
    """, (day,))
    present = [row for row in rows if row.get("final_status") == "PRESENT"]
    absent = [row for row in rows if row.get("final_status") == "ABSENT"]
    if "absent" in text:
        names = ", ".join(row.get("name") or "Unknown" for row in absent) or "No absent records"
        return f"Attendance for {day}: {len(absent)} absent. {names}."
    if "present" in text or "how many" in text:
        return f"Attendance for {day}: {len(present)} present out of {enrolled} enrolled."
    if "late" in text:
        late = [row for row in present if row.get("first_seen") and row["first_seen"] > "09:10"]
        names = ", ".join(row.get("name") or "Unknown" for row in late) or "No late arrivals recorded"
        return f"Late arrivals for {day}: {len(late)}. {names}."
    return (f"Local attendance assistant for {day}: {len(present)} present, "
            f"{len(absent)} absent, {enrolled} enrolled. Add GEMINI_API_KEY to .env "
            "for full natural-language answers.")


# ── Status ────────────────────────────────────────────────────────────────────

@ai_router.get("/status")
def ai_status():
    server_key = bool(os.getenv("GEMINI_API_KEY", ""))
    return {
        "server_key":  server_key,
        "demo_mode":   not server_key,
        "engine_ready": _engine is not None,
        "model":       "gemini-2.5-flash",
        "priority_order": [
            "chatbot", "dataset", "analytics", "report", "debug"
        ],
        "debug_types": ["facetrack", "sqlite", "rtsp", "general"],
    }


# ── #1 Chatbot ────────────────────────────────────────────────────────────────

@ai_router.post("/chat")
async def chat(req: ChatReq,
                x_demo_key: Optional[str] = Header(default=None)):
    """
    Single-turn, non-streaming — matches the dashboard's chat box exactly:
    sends {message, date}, expects one JSON {answer} back.
    """
    try:
        key = x_demo_key or os.getenv("GEMINI_API_KEY", "")
        if not key:
            return {"answer": local_chat_answer(req.message, req.date)}
        answer = await eng().chat_sync(req.message, work_date=req.date, demo_key=x_demo_key)
        return {"answer": answer}
    except Exception as exc:
        logger.exception("AI chat request failed")
        return JSONResponse(
            status_code=500,
            content={"detail": f"AI chat failed: {exc}"},
        )


# ── Suggested questions (dashboard chip list) ──────────────────────────────────

SUGGESTED_QUESTIONS = [
    "Who is absent today?", "How many faculty are present?",
    "Show late arrivals today", "Who is on leave this week?",
    "Monthly attendance summary", "Which faculty has lowest attendance?",
    "How many permissions this month?", "Download today's CSV",
    "Who is absent more than 3 days?", "Give full attendance report",
]

@ai_router.get("/suggestions")
def suggestions():
    return {"suggestions": SUGGESTED_QUESTIONS}


# ── #2 Dataset Analysis ───────────────────────────────────────────────────────

@ai_router.post("/dataset")
async def dataset(req: DatasetReq,
                   x_demo_key: Optional[str] = Header(default=None)):
    return sse(eng().analyze_dataset(req.days, req.focus, demo_key=x_demo_key))


# ── #3 Analytics ──────────────────────────────────────────────────────────────

@ai_router.post("/analytics")
async def analytics(req: AnalyticsReq,
                     x_demo_key: Optional[str] = Header(default=None)):
    return sse(eng().analytics(req.question, req.work_date, demo_key=x_demo_key))


# ── #4 Report ─────────────────────────────────────────────────────────────────

@ai_router.post("/report")
async def report(req: ReportReq,
                  x_demo_key: Optional[str] = Header(default=None)):
    return sse(eng().generate_report(
        req.from_date, req.to_date, req.report_type, demo_key=x_demo_key))


# ── #5 Debug ──────────────────────────────────────────────────────────────────

@ai_router.post("/debug")
async def debug(req: DebugReq,
                 x_demo_key: Optional[str] = Header(default=None)):
    return sse(eng().debug(
        req.error_msg, req.traceback, req.code_snippet,
        req.debug_type, demo_key=x_demo_key))
