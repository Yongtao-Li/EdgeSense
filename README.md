# EdgeSense
Open-source video-based ski coaching using pose estimation and biomechanics.

EdgeSense analyzes skiing videos to identify turn structure, balance, outside-ski dominance, and hip angulation. Using pose estimation and rule-based biomechanics, it produces annotated visual feedback and clear coaching cues—locally, privately, and explainably.

Core differentiators

📷 Camera-only (no sensors)

🧠 Explainable rules (not black-box scores)

⛷️ Ski-specific turn intelligence

🔒 Local-first, privacy-respecting

☁️ Cloud-ready for real-time coaching

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

## Run (CLI)

```bash
skicoach analyze --input path/to/video.mp4 --output data/outputs --pose mediapipe
```

Outputs are written to `data/outputs/<video_name>/`:

- `report.json`
- `report.md`
- `annotated_preview.mp4` (browser-optimized playback)

## Run (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

## Configuration

Tune thresholds in `skicoach/config.py`:

- `MIN_VISIBILITY`
- `SMOOTHING_WINDOW`
- `OUTSIDE_SKI_OFFSET_THRESHOLD`
- `COM_CENTERED_RATIO_THRESHOLD`
- `ANGULATION_DEG_THRESHOLD`
- `LEG_TILT_MIN_DEG`
- `LATE_COM_RATIO_THRESHOLD`
- `TURN_MIN_FRAMES`
- `CROP_MARGIN_X`
- `CROP_MARGIN_Y`
- `PERSON_DETECTOR_BACKEND`
- `YOLO_MODEL`
- `YOLO_CONF_THRESHOLD`
- `YOLO_IOU_THRESHOLD`
- `RENDER_MAX_UPSCALE`
- `RENDER_SIZE_PERCENTILE`
- `RENDER_MIN_CROP_WIDTH`
- `RENDER_MIN_CROP_HEIGHT`
- `RENDER_WARMUP_FRAMES`
- `RENDER_WARMUP_SCALE`
- `POSE_CROP_MARGIN_X`
- `POSE_CROP_MARGIN_Y`
- `POSE_DRIFT_MAX_CENTER_DIST_NORM`
- `POSE_DRIFT_HOLD_FRAMES`
- `TRACK_MATCH_SCORE_THRESHOLD`
- `TRACK_MISS_STREAK_TO_LOST`
- `TRACK_HIT_STREAK_TO_LOCK`
- `VIDEO_CODEC_CANDIDATES`

For higher-quality multi-person detection, install YOLO runtime:

```bash
pip install -e ".[yolo]"
```

## Tests

```bash
pytest -q
```

## MVP Todo List

### Week 1: Reliability + Core UX

- [x] Add target skier selection UI for first-frame person pick.
- [x] Add target tracking lock and reacquire behavior for multi-person scenes.
- [x] Run pose on tracked target crop and map keypoints back to frame space.
- [ ] Add per-turn timeline with init/apex/finish jump controls in Streamlit.
- [ ] Add per-turn confidence scoring from visibility and tracking stability.

### Week 2: Coaching Depth + Progress

- [ ] Add threshold tuning panel in Streamlit with reset-to-default.
- [ ] Add left-vs-right asymmetry dashboard for outside-ski, angulation, and timing.
- [ ] Add local session history and progression trends.
- [ ] Harden with tests for tracking lock, confidence bounds, and rule regressions.

### Priority Order

- [ ] Selection + tracking lock
- [ ] Turn navigation + confidence
- [ ] Threshold tuning
- [ ] Asymmetry + trends
- [ ] Test hardening
