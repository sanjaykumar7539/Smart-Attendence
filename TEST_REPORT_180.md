# Adaptive Recognition — 180-case robustness report

## Scope and result
180/180 deterministic synthetic tests passed after correcting the test harness. The first run had 10 false failures because it expected super-resolution on good-quality 112-pixel faces; the corrected expectation is that SR is skipped. The original six adaptive tests and two quality smoke tests also passed. Python compilation and dashboard JavaScript/ID preservation checks passed.

## Coverage
- 120 image-condition cases: five in-plane rotations (-60, -30, 0, 30, 60 degrees), three crop sizes (32, 64, 112 pixels), two blur levels (Gaussian sigma 0 and 3), and four brightness multipliers (0.12, 0.35, 0.7, 1.0).
- 60 mocked recognition-decision cases: confirmation quorum, low-margin rejection, mixed identities, AdaFace disagreement, SR routing/call limit, and unknown identity.
- Synthetic images are random textured patches, not photographs of people. Recognition embeddings and SR/AdaFace outputs are mocked. In-plane rotation does not test real 3D head yaw or profile faces.

## Important limitations
These results demonstrate code-path robustness, not biometric accuracy. No real 159-student dataset, enrollment gallery, classroom video, or compatible SR/AdaFace weights were available for this run. No real false-acceptance rate, false-rejection rate, FPS, or near/middle/far accuracy is claimed. The ZIP includes the full per-case JSON and reproducible test script.

## Findings requiring real-world validation
The current quality analyzer uses a simple landmark pose heuristic and does not estimate full 3D head pose. SR may create visually plausible but identity-inaccurate detail. The existing adaptive manager does not compare original-versus-enhanced recognition scores before accepting the enhanced embedding. AdaFace agreement and score fusion require calibration on a held-out student dataset. The original attendance policy still contains staff-oriented working-day rules; student timetable/period rules must be configured before production use. The current project should not be considered production-validated merely because synthetic tests pass.

## Real-data acceptance plan
With consent and authorized student enrollment, evaluate at least 159 distinct identities using separate enrollment and test sessions. Include frontal, left/right yaw, pitch, low light, backlight, motion blur, small faces, glasses/occlusion, and unknown visitors. Compare ArcFace-only against adaptive ArcFace, SR, and AdaFace separately. Report detection recall, identification accuracy, false acceptance/rejection, unknown rejection, per-distance results, per-stage latency, and attendance-event correctness. Do not tune thresholds on the final test set.
