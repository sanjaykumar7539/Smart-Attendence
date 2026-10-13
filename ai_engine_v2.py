"""
ai_engine_v2.py
────────────────
AI Engine v2 — updated per priorities:
  #1 Admin Chatbot        — richest context, multi-turn, full system awareness
  #2 Dataset Analysis     — 30/60/90 day patterns, anomaly detection, health score
  #3 Attendance Analytics — NL→SQL→insight, live data
  #4 Auto Report          — structured report generation
  #5 Code Debug           — ALL 4 types: FaceTrack errors, SQLite, RTSP, General Python

Backend: Google Gemini API (gemini-2.5-flash) — switched from Anthropic
2026-07-28 for cost reasons (Gemini has a genuine recurring free tier,
no expiry; Anthropic's is a one-time trial credit).

Dual-mode API key:
  - Server mode:  reads from GEMINI_API_KEY env var (production)
  - Browser mode: key passed per-request (demo / standalone use)
               endpoint: /api/ai/* with header X-Demo-Key: AIza...
               Get a key free at https://aistudio.google.com/apikey
"""

import os
import json
import sqlite3
import logging
import httpx
from datetime import date, datetime, timedelta
from typing import AsyncGenerator, List, Dict, Any, Optional

logger = logging.getLogger(__name__)

GEMINI_MODEL    = "gemini-2.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class AIEngine:
    def __init__(self,
                 db_path:          str = "attendance.db",
                 hlc_team_db:      str = "hlc_team.db",
                 hlc_visitor_db:   str = "hlc_visitors.db",
                 api_key:          str = None):
        self.db_path        = db_path
        self.hlc_team_db    = hlc_team_db
        self.hlc_visitor_db = hlc_visitor_db
        self._server_key    = api_key or os.getenv("GEMINI_API_KEY", "")

    # ── Key resolution (server key takes precedence) ──────────────────────────
    def _key(self, demo_key: str = None) -> str:
        return self._server_key or demo_key or ""

    # ── DB helpers ────────────────────────────────────────────────────────────
    def _query(self, db_path: str, sql: str, params: tuple = ()) -> List[Dict]:
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning(f"DB query ({db_path}): {e}")
            return []

    def _scalar(self, db_path: str, sql: str, params: tuple = ()) -> Any:
        rows = self._query(db_path, sql, params)
        return list(rows[0].values())[0] if rows else None

    # ─────────────────────────────────────────────────────────────────────────
    # CONTEXT BUILDERS
    # ─────────────────────────────────────────────────────────────────────────

    # ── #1 Chatbot context — richest, most live ───────────────────────────────
    def _ctx_chatbot(self, work_date: str = None) -> str:
        today = work_date or date.today().isoformat()

        # Live attendance
        att_today = self._query(self.db_path, """
            SELECT a.employee_id, e.name, e.department AS dept,
                   a.first_seen, a.last_seen, a.final_status
            FROM attendance a
            LEFT JOIN faculty e ON a.employee_id = e.employee_id
            WHERE a.work_date = ?
            ORDER BY a.first_seen
        """, (today,))

        enrolled  = self._scalar(self.db_path, "SELECT COUNT(*) FROM faculty") or 0
        present   = sum(1 for r in att_today if r.get("final_status") == "PRESENT")
        absent    = sum(1 for r in att_today if r.get("final_status") == "ABSENT")
        perm      = sum(1 for r in att_today if r.get("final_status") == "PERMISSION")
        half      = sum(1 for r in att_today if r.get("final_status") == "HALF_DAY")

        # Department breakdown
        dept_stats = self._query(self.db_path, """
            SELECT e.department AS dept,
                   COUNT(DISTINCT a.employee_id) as faculty,
                   SUM(CASE WHEN a.final_status='PRESENT' THEN 1 ELSE 0 END) as present
            FROM attendance a
            LEFT JOIN faculty e ON a.employee_id = e.employee_id
            WHERE a.work_date = ?
            GROUP BY e.department ORDER BY faculty DESC
        """, (today,))

        # Recent 5 days trend
        trend = self._query(self.db_path, """
            SELECT work_date,
                   SUM(CASE WHEN final_status='PRESENT' THEN 1 ELSE 0 END) as present,
                   COUNT(DISTINCT employee_id) as tracked
            FROM attendance
            WHERE work_date >= date(?, '-5 days')
            GROUP BY work_date ORDER BY work_date DESC
            """, (today,))

        week_end = (date.fromisoformat(today) + timedelta(days=6)).isoformat()
        approved_leave = self._query(self.db_path, """
            SELECT l.employee_id, e.name, l.leave_type,
                   l.from_date, l.to_date, l.days, l.reason
            FROM leave_applications l
            LEFT JOIN faculty e ON e.employee_id=l.employee_id
            WHERE l.status='APPROVED'
              AND l.from_date <= ? AND l.to_date >= ?
            ORDER BY l.from_date, e.name
        """, (week_end, today))

        # HLC live data
        hlc_section = ""
        try:
            hlc_inside = self._query(self.hlc_visitor_db, """
                SELECT visitor_id, name, dept, entry_time,
                    ROUND((julianday('now','localtime') -
                           julianday(visit_date||'T'||entry_time))*1440,1) AS mins_inside
                FROM visitor_sessions
                WHERE visit_date=? AND exit_time IS NULL
                ORDER BY entry_time
            """, (today,))
            hlc_team_present = self._query(self.hlc_team_db, """
                SELECT a.employee_id, e.name, e.hlc_role, a.first_seen
                FROM hlc_attendance a
                JOIN hlc_employees e ON a.employee_id = e.employee_id
                WHERE a.work_date=? AND a.final_status IN ('PRESENT','PERMISSION')
            """, (today,))
            open_alerts = self._scalar(self.hlc_visitor_db,
                "SELECT COUNT(*) FROM visit_alerts WHERE status='OPEN' AND visit_date=?",
                (today,)) or 0
            hlc_section = f"""
HLC (Horizon Learning Center):
  Team present: {len(hlc_team_present)}
  Visitors currently inside: {len(hlc_inside)}
  Open alerts (>15min overstay): {open_alerts}
  Currently inside: {json.dumps(hlc_inside, default=str)}"""
        except Exception:
            hlc_section = "HLC: data not available"

        return f"""=== FACETRACK PRO — LIVE SYSTEM CONTEXT ===
Institution: EGS Pillay Engineering College, Nagapattinam
Date: {today}  |  Time: {datetime.now().strftime('%H:%M')}

TODAY'S ATTENDANCE:
  Enrolled: {enrolled}  |  Present: {present}  |  Absent: {absent}
  Permission: {perm}  |  Half Day: {half}
  Attendance %: {round(present/(enrolled or 1)*100,1)}%

DEPARTMENT BREAKDOWN:
{json.dumps(dept_stats, default=str)}

FULL ATTENDANCE LIST (today):
{json.dumps(att_today, default=str)}

5-DAY TREND:
{json.dumps(trend, default=str)}

APPROVED LEAVE THIS WEEK ({today} to {week_end}):
{json.dumps(approved_leave, default=str)}

{hlc_section}

SYSTEM INFO:
  Hardware: NVIDIA Jetson Orin Nano Super, JetPack R36.4.7
  Model: InsightFace buffalo_s · FAISS-CPU · SORT tracker
  Camera: HikVision DS-2CD3151G0-I (192.168.0.100)
  Enrolled faces: 7 (Sriganesh, Sakthivel, Sundar, Vishal, Sanjay, Annamala +1)

ATTENDANCE POLICY:
  Present: arrival ≤ 09:05 | Permission: 09:05–09:55 (max 2/month)
  Half Day: departure < 12:45 | CL: 12/yr | ML: 12/yr | EL: 30/yr

DB SCHEMA (for SQL queries):
  faculty(employee_id, name, department, designation, staff_category, is_hlc)
  attendance(employee_id, work_date, first_seen, last_seen,
             morning_status, evening_status, final_status, notes)
  detections(id, employee_id, timestamp, direction, confidence, camera_id)"""

    # ── #2 Dataset context ────────────────────────────────────────────────────
    def _ctx_dataset(self, days: int = 30, focus: str = "") -> str:
        end   = date.today().isoformat()
        start = (date.today() - timedelta(days=days)).isoformat()

        daily = self._query(self.db_path, """
            SELECT work_date,
                   COUNT(DISTINCT employee_id) as tracked,
                   SUM(CASE WHEN final_status='PRESENT'    THEN 1 ELSE 0 END) as present,
                   SUM(CASE WHEN final_status='ABSENT'     THEN 1 ELSE 0 END) as absent,
                   SUM(CASE WHEN final_status='PERMISSION' THEN 1 ELSE 0 END) as permission,
                   SUM(CASE WHEN final_status='HALF_DAY'   THEN 1 ELSE 0 END) as half_day
            FROM attendance
            WHERE work_date BETWEEN ? AND ?
            GROUP BY work_date ORDER BY work_date
        """, (start, end))

        per_employee = self._query(self.db_path, """
            SELECT e.name, e.department AS dept,
                   COUNT(*) as days_recorded,
                   SUM(CASE WHEN a.final_status='PRESENT'    THEN 1 ELSE 0 END) as present,
                   SUM(CASE WHEN a.final_status='ABSENT'     THEN 1 ELSE 0 END) as absent,
                   SUM(CASE WHEN a.final_status='PERMISSION' THEN 1 ELSE 0 END) as permission,
                   ROUND(SUM(CASE WHEN a.final_status='PRESENT' THEN 1.0 ELSE 0 END) /
                         NULLIF(COUNT(*),0)*100, 1) as attendance_pct,
                   MIN(a.first_seen) as earliest_arrival,
                   MAX(a.first_seen) as latest_arrival
            FROM attendance a
            LEFT JOIN faculty e ON a.employee_id=e.employee_id
            WHERE a.work_date BETWEEN ? AND ?
            GROUP BY a.employee_id ORDER BY absent DESC
        """, (start, end))

        dept_summary = self._query(self.db_path, """
            SELECT e.department AS dept,
                   COUNT(DISTINCT a.employee_id) as faculty_count,
                   ROUND(AVG(CASE WHEN a.final_status='PRESENT' THEN 1.0 ELSE 0 END)*100,1)
                       as avg_attendance_pct,
                   SUM(CASE WHEN a.final_status='ABSENT' THEN 1 ELSE 0 END) as total_absences
            FROM attendance a
            LEFT JOIN faculty e ON a.employee_id=e.employee_id
            WHERE a.work_date BETWEEN ? AND ?
            GROUP BY e.department ORDER BY avg_attendance_pct ASC
        """, (start, end))

        # Weekly patterns (Mon–Fri)
        weekly = self._query(self.db_path, """
            SELECT strftime('%w', work_date) as weekday,
                   ROUND(AVG(CASE WHEN final_status='PRESENT' THEN 1.0 ELSE 0 END)*100,1)
                       as avg_attendance
            FROM attendance
            WHERE work_date BETWEEN ? AND ?
            GROUP BY weekday ORDER BY weekday
        """, (start, end))

        return f"""=== DATASET ANALYSIS CONTEXT ===
Institution: EGS Pillay Engineering College, Nagapattinam
Period: {start} → {end} ({days} days)
Focus: {focus or "full analysis"}

DAILY TREND ({len(daily)} working days):
{json.dumps(daily, default=str)}

PER-EMPLOYEE SUMMARY:
{json.dumps(per_employee, default=str)}

DEPARTMENT SUMMARY:
{json.dumps(dept_summary, default=str)}

WEEKDAY PATTERNS (0=Sun, 1=Mon … 6=Sat):
{json.dumps(weekly, default=str)}"""

    # ── #3 Analytics context ──────────────────────────────────────────────────
    def _ctx_analytics(self, work_date: str = None) -> str:
        d = work_date or date.today().isoformat()
        att = self._query(self.db_path, """
            SELECT a.employee_id, e.name, e.department AS dept,
                   a.first_seen, a.last_seen, a.final_status, a.notes
            FROM attendance a
            LEFT JOIN faculty e ON a.employee_id=e.employee_id
            WHERE a.work_date=? ORDER BY a.first_seen
        """, (d,))
        enrolled = self._scalar(self.db_path, "SELECT COUNT(*) FROM faculty") or 0
        trend = self._query(self.db_path, """
            SELECT work_date,
                   SUM(CASE WHEN final_status='PRESENT' THEN 1 ELSE 0 END) as present,
                   COUNT(*) as total
            FROM attendance
            WHERE work_date >= date(?,'-7 days') AND work_date<=?
            GROUP BY work_date ORDER BY work_date
        """, (d, d))
        return f"""=== ATTENDANCE DATA — {d} ===
Enrolled: {enrolled}
Today records: {len(att)}

ATTENDANCE RECORDS:
{json.dumps(att, default=str)}

7-DAY TREND:
{json.dumps(trend, default=str)}

SCHEMA:
  faculty(employee_id, name, department, designation, staff_category, is_hlc)
  attendance(employee_id, work_date, first_seen, last_seen,
             morning_status, evening_status, final_status, notes)
  detections(id, employee_id, timestamp, direction, confidence, camera_id)"""

    # ── #4 Report context ─────────────────────────────────────────────────────
    def _ctx_report(self, from_date: str, to_date: str,
                     report_type: str = "summary") -> str:
        emp_summary = self._query(self.db_path, """
            SELECT e.name, e.department AS dept,
                   COUNT(*) as total_days,
                   SUM(CASE WHEN a.final_status='PRESENT'    THEN 1 ELSE 0 END) as present,
                   SUM(CASE WHEN a.final_status='ABSENT'     THEN 1 ELSE 0 END) as absent,
                   SUM(CASE WHEN a.final_status='PERMISSION' THEN 1 ELSE 0 END) as permission,
                   SUM(CASE WHEN a.final_status='HALF_DAY'   THEN 1 ELSE 0 END) as half_day,
                   ROUND(SUM(CASE WHEN a.final_status='PRESENT' THEN 1.0 ELSE 0 END)/
                         NULLIF(COUNT(*),0)*100,1) as pct
            FROM attendance a
            LEFT JOIN faculty e ON a.employee_id=e.employee_id
            WHERE a.work_date BETWEEN ? AND ?
            GROUP BY a.employee_id ORDER BY absent DESC
        """, (from_date, to_date))
        return f"""=== REPORT DATA ===
Period: {from_date} to {to_date}  |  Type: {report_type}
PER-EMPLOYEE: {json.dumps(emp_summary, default=str)}"""

    # ── #5 Debug context — ALL 4 TYPES ────────────────────────────────────────
    def _ctx_debug(self, error_msg: str, traceback: str,
                    code_snippet: str, debug_type: str) -> str:

        type_contexts = {
            "facetrack": """
FACETRACK PRO CODEBASE:
  main.py          — FastAPI app, WebSocket, RTSP stream handler
  recognition.py   — InsightFace buffalo_s, FAISS-CPU, ArcFace R100
  tracker.py       — SORT tracker, min_hits=2, direction detection
  camera_reader.py — HikVision RTSP, OpenCV, CLAHE preprocessing
  attendance.py    — policy logic, morning/evening status, vote buffer
  config.py        — detection_threshold=0.20, recognition_threshold=0.22
  audit_engine.py  — CLEAN/CORRECTED/FLAGGED/ABSENT_AUTO stamping
  hlc_processor_v4.py — multi-door RoomStateManager

KNOWN ISSUES:
  - numpy MUST be 1.26.4 (2.x breaks onnxruntime on aarch64)
  - cuda-python MUST be 12.6.0
  - InsightFace TRT patches lost if package reinstalled
  - buffalo_l vs buffalo_s — latency trade-off
  - VoteBuffer race condition on state.latest_frame""",

            "sqlite": """
SQLITE SETUP:
  DB files: attendance.db, hlc_team.db, hlc_visitors.db
  All use WAL journal mode, foreign_keys=ON
  Thread-local connections (threading.local pattern)
  Python sqlite3 stdlib — no SQLAlchemy

COMMON SQLITE ISSUES ON JETSON:
  - "database is locked" → multiple threads writing, WAL not enabled
  - "no such table" → migration not run, check _init_tables()
  - "UNIQUE constraint failed" → duplicate upsert, use INSERT OR REPLACE
  - Slow queries → missing index, use EXPLAIN QUERY PLAN
  - Corrupted DB → WAL file not checkpointed after crash""",

            "rtsp": """
CAMERA SETUP:
  Main cam: HikVision DS-2CD3151G0-I
  IP: 192.168.0.100  RTSP: rtsp://admin:Admin%40123@192.168.0.100:554/Streaming/Channels/102
  HLC cam(s): 192.168.0.101–104
  Jetson IP: 192.168.0.109 (WiFi) / 192.168.0.112 (Ethernet)
  OpenCV 4.9.0 + GStreamer backend

RTSP TROUBLESHOOTING:
  - "empty frame" → wrong channel suffix (try /101 vs /102)
  - "%40" in password = "@" URL-encoded
  - GStreamer pipeline: use cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
  - Frame drops: frame_skip=4 in config.py
  - Reconnect: destroy and re-create VideoCapture on consecutive empty frames
  - HikVision substream: /Streaming/Channels/102 (lower res, lower latency)
  - Network: camera and Jetson must be on SAME subnet""",

            "general": """
ENVIRONMENT:
  OS: Ubuntu 22.04 (aarch64)  Hardware: Jetson Orin Nano Super
  JetPack: R36.4.7  CUDA: 12.6  TensorRT: 10.3.0
  Python: 3.10  FastAPI: 0.111  uvicorn with asyncio loop
  Working dir: ~/Desktop/facetrack (NOT facetrack_demo)

GENERAL PYTHON ISSUES:
  - Import errors → check venv is activated, correct Python version
  - Async/await in sync context → use asyncio.run() or loop.run_until_complete()
  - Memory leak → check for unclosed DB connections or CV2 captures
  - Threading issues → use threading.local for DB connections
  - Pydantic v2 breaking changes → field validators syntax changed"""
        }

        ctx = type_contexts.get(debug_type, type_contexts["general"])

        return f"""=== FACETRACK PRO DEBUG ===
Error Type: {debug_type.upper()}
{ctx}

ERROR:
{error_msg}

TRACEBACK:
{traceback or "(not provided)"}

CODE:
{code_snippet or "(not provided)"}"""

    # ─────────────────────────────────────────────────────────────────────────
    # SYSTEM PROMPTS
    # ─────────────────────────────────────────────────────────────────────────

    SYSTEM_PROMPTS = {
        "chatbot": """You are the AI admin assistant for FaceTrack Pro at EGS Pillay Engineering College, Nagapattinam.
You have live access to today's attendance data, HLC visitor logs, and system status.
Respond conversationally but precisely. Reference actual names and numbers from the data.
For SQL requests, write valid SQLite. For policy questions, apply EGS Pillay rules exactly.
Keep replies concise unless detail is requested. Be helpful, not bureaucratic.""",

        "dataset": """You are a data analyst specialising in educational institution attendance patterns.
Provide a structured analysis with these exact sections:
1. Overall Health Score (A–F grade, explain why)
2. Key Findings (top 5 bullet points with specific numbers)
3. Anomalies & Red Flags (anything unusual or concerning)
4. Weekday Patterns (which days are worst/best and why)
5. Department Comparison (ranked table)
6. Chronic Absentees (anyone >20% absent — name them)
7. Actionable Recommendations (3 specific, implementable steps)
Be data-driven. Cite actual numbers. Be direct about problems.""",

        "analytics": """You are an attendance analytics AI for EGS Pillay Engineering College.
Answer questions precisely using the provided attendance data.
For SQL requests: write valid SQLite using the exact schema provided.
Format numbers clearly. Use tables for comparisons. Be concise.""",

        "report": """You are generating a formal attendance report for EGS Pillay Engineering College management.
Structure: Executive Summary → Key Metrics table → Department Analysis →
Individual Performance → Concerns & Recommendations → Conclusion.
Use markdown tables. Be professional and data-driven.
Flag anyone with <75% attendance as a concern.""",

        "debug_facetrack": """You are a senior engineer who built FaceTrack Pro — an InsightFace + FAISS + SORT
facial recognition attendance system on NVIDIA Jetson Orin Nano.
Diagnose the error and provide:
1. Root cause (2 sentences max)
2. Exact fix (code if needed)
3. One-line prevention tip
Be direct and technical. No fluff.""",

        "debug_sqlite": """You are a SQLite expert familiar with Python's sqlite3 module and multi-threaded applications.
Diagnose the database error and provide:
1. Root cause
2. Exact fix with code
3. Prevention / best practice
Consider WAL mode, threading, and connection management.""",

        "debug_rtsp": """You are an expert in IP cameras, RTSP streams, OpenCV, and GStreamer on embedded Linux.
Diagnose the camera/stream issue and provide:
1. Root cause
2. Step-by-step fix
3. Verification command
Consider HikVision-specific quirks, URL encoding, and Jetson network config.""",

        "debug_general": """You are a senior Python engineer specialising in FastAPI, async code,
and embedded Linux (aarch64 / Jetson).
Diagnose the error precisely and provide a working fix.""",
    }

    # ─────────────────────────────────────────────────────────────────────────
    # CLAUDE STREAMING CALL
    # ─────────────────────────────────────────────────────────────────────────

    async def stream(self, messages: List[Dict], system: str,
                      max_tokens: int = 2048,
                      demo_key: str = None) -> AsyncGenerator[str, None]:
        key = self._key(demo_key)
        if not key:
            yield ("⚠ No API key. Add GEMINI_API_KEY to .env (server) or "
                   "enter one in Settings (demo). Free key: "
                   "https://aistudio.google.com/apikey")
            return

        # Gemini uses "model" for the assistant role, not "assistant", and
        # wraps text in a parts[] list rather than a plain content string.
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        payload = {
            "contents":         contents,
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        url = f"{GEMINI_BASE_URL}/{GEMINI_MODEL}:streamGenerateContent"

        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                async with client.stream(
                    "POST", url,
                    params={"key": key, "alt": "sse"},
                    headers={"content-type": "application/json"},
                    json=payload,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        try:
                            err = json.loads(body).get("error", {})
                            msg = err.get("message", body.decode()[:200])
                        except Exception:
                            msg = body.decode()[:200]
                        yield f"Gemini API error {resp.status_code}: {msg}"
                        return
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if not data or data == "[DONE]":
                            continue
                        try:
                            obj = json.loads(data)
                            candidates = obj.get("candidates", [])
                            if not candidates:
                                continue
                            parts = candidates[0].get("content", {}).get("parts", [])
                            for p in parts:
                                text = p.get("text", "")
                                if text:
                                    yield text
                        except json.JSONDecodeError:
                            pass
        except httpx.TimeoutException:
            yield "\n\n⚠ Request timed out. Try a shorter question."
        except Exception as e:
            yield f"\n\n⚠ Error: {e}"

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE METHODS (priority order)
    # ─────────────────────────────────────────────────────────────────────────

    # #1 Chatbot
    async def chatbot(self, messages: List[Dict],
                       demo_key: str = None) -> AsyncGenerator[str, None]:
        ctx = self._ctx_chatbot()
        augmented = []
        for i, m in enumerate(messages):
            if i == 0 and m["role"] == "user":
                augmented.append({"role": "user",
                                    "content": f"{ctx}\n\n---\n{m['content']}"})
            else:
                augmented.append(m)
        async for chunk in self.stream(augmented,
                                        self.SYSTEM_PROMPTS["chatbot"],
                                        max_tokens=1500,
                                        demo_key=demo_key):
            yield chunk

    # #1b Chatbot — single-turn, non-streaming (used by the dashboard's chat box)
    async def chat_sync(self, message: str, work_date: str = None,
                         demo_key: str = None) -> str:
        """
        Same context/system-prompt as chatbot() above, but collects the full
        streamed response into one string and returns it — matches the
        dashboard's {message, date} -> {answer} contract (no SSE needed).
        """
        ctx  = self._ctx_chatbot(work_date)
        msgs = [{"role": "user", "content": f"{ctx}\n\n---\n{message}"}]
        parts = []
        async for chunk in self.stream(msgs, self.SYSTEM_PROMPTS["chatbot"],
                                        max_tokens=1500, demo_key=demo_key):
            parts.append(chunk)
        return "".join(parts)

    # #2 Dataset Analysis
    async def analyze_dataset(self, days: int = 30, focus: str = "",
                               demo_key: str = None) -> AsyncGenerator[str, None]:
        ctx    = self._ctx_dataset(days, focus)
        prompt = (f"Analyse the attendance dataset for the last {days} days."
                  + (f" Focus: {focus}" if focus else "")
                  + " Provide the complete structured analysis.")
        msgs = [{"role": "user", "content": f"{ctx}\n\n{prompt}"}]
        async for chunk in self.stream(msgs, self.SYSTEM_PROMPTS["dataset"],
                                        max_tokens=3000, demo_key=demo_key):
            yield chunk

    # #3 Analytics
    async def analytics(self, question: str, work_date: str = None,
                         demo_key: str = None) -> AsyncGenerator[str, None]:
        ctx  = self._ctx_analytics(work_date)
        msgs = [{"role": "user", "content": f"{ctx}\n\nQuestion: {question}"}]
        async for chunk in self.stream(msgs, self.SYSTEM_PROMPTS["analytics"],
                                        max_tokens=2000, demo_key=demo_key):
            yield chunk

    # #4 Report
    async def generate_report(self, from_date: str, to_date: str,
                               report_type: str = "monthly",
                               demo_key: str = None) -> AsyncGenerator[str, None]:
        ctx    = self._ctx_report(from_date, to_date, report_type)
        prompt = f"Generate a complete {report_type} attendance report for {from_date} to {to_date}."
        msgs   = [{"role": "user", "content": f"{ctx}\n\n{prompt}"}]
        async for chunk in self.stream(msgs, self.SYSTEM_PROMPTS["report"],
                                        max_tokens=3500, demo_key=demo_key):
            yield chunk

    # #5 Debug — all 4 types
    async def debug(self, error_msg: str, traceback: str = "",
                    code_snippet: str = "",
                    debug_type: str = "general",
                    demo_key: str = None) -> AsyncGenerator[str, None]:
        ctx    = self._ctx_debug(error_msg, traceback, code_snippet, debug_type)
        system = self.SYSTEM_PROMPTS.get(f"debug_{debug_type}",
                                          self.SYSTEM_PROMPTS["debug_general"])
        msgs   = [{"role": "user",
                    "content": f"{ctx}\n\nDiagnose and fix this error."}]
        async for chunk in self.stream(msgs, system,
                                        max_tokens=2000, demo_key=demo_key):
            yield chunk
