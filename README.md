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
- `annotated.mp4`

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
- `VIDEO_CODEC_CANDIDATES`

## Tests

```bash
pytest -q
```

## MVP Todo List

### Week 1: Reliability + Core UX

- [ ] Add target skier selection UI for first-frame person pick.
- [ ] Add target tracking lock and reacquire behavior for multi-person scenes.
- [ ] Run pose on tracked target crop and map keypoints back to frame space.
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
