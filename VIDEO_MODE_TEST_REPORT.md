# Recorded Video Mode Test Report

## Result
- Python compile: PASS
- Recorded MP4 frame-read smoke test: PASS
- Adaptive pipeline smoke test: PASS
- Adaptive recognition cases: 6/6 PASS
- Robustness suite: 180/180 PASS

## Video smoke test
A temporary 320x240, 10 FPS, 12-frame MP4 was generated locally. `CameraReader(input_mode="video")` successfully opened it and returned a valid `(240, 320, 3)` BGR frame while preserving real-time playback pacing.

## Supported source modes
- `rtsp`
- `webcam`
- `video`

## Real classroom limitation
This verifies software video-source handling and existing recognition decision logic. It does not measure real student recognition accuracy. For that, use your enrolled gallery and real recorded classroom video.
