# Recorded Video Test Mode

Use this mode to test the Smart Attendance pipeline with a saved classroom video.

## 1. Put the video in the project folder
Example:

`testing.mp4`

## 2. Edit `config.py`
Under `CameraConfig` set:

```python
input_mode = "video"
video_path = "./testing.mp4"
video_loop = False
video_realtime = True
```

Available source modes:
- `rtsp` — IP/CCTV stream
- `webcam` — USB/laptop camera
- `video` — MP4/AVI/MOV file supported by OpenCV/FFmpeg

`video_realtime=True` preserves the recorded video FPS, which is recommended for tracking, temporal voting and attendance-flow testing.

## 3. Run normally

```bash
python main.py
```

Then open the dashboard at the same FastAPI URL used by the project.

## Important
This ZIP does not include a classroom test video or optional Real-ESRGAN/AdaFace model weights. Real recognition accuracy depends on your enrolled gallery, video quality, pose, face pixels, lighting and installed model files.
