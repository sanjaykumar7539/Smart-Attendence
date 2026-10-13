# Adaptive Smart Attendance Test Report

Date: 2026-09-09

## Environment validation
- Python compileall: PASS
- Original adaptive smoke test: PASS
- Adaptive decision/edge-case suite: 6/6 PASS

## Test cases
1. Confirm after temporal vote quorum — PASS
2. Reject low Top-1/Top-2 margin as UNKNOWN — PASS
3. Keep identity UNCERTAIN before vote quorum — PASS
4. Reject ArcFace/AdaFace identity disagreement — PASS
5. Run Super Resolution at most once per track (configured limit) — PASS
6. Clean buffers/votes for inactive tracks — PASS

## Bug found and fixed during testing
`adaface_verifier.py` imported FAISS unconditionally. This caused the adaptive module to fail during import even when AdaFace was disabled and FAISS was unavailable. FAISS is now optional inside the AdaFace verifier; AdaFace falls back cleanly when its dependencies/models are unavailable.

## Scope limitation
These tests validate control flow, quality gating, decision logic, voting, SR trigger control, verifier disagreement safety, cleanup and syntax. They do not measure real-world face recognition accuracy because no labeled evaluation face set and compatible SR/AdaFace model weights were bundled with this package. Real accuracy/FRR/FAR testing requires enrollment images/videos and the actual deployment model files.
