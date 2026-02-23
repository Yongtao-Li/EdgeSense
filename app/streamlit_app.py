import base64
import tempfile
from pathlib import Path

import cv2
import streamlit as st

from skicoach.pipeline import analyze_video
from skicoach.pose_provider import PersonCandidate, active_person_detector_backend, detect_person_candidates_first_frame


def _read_first_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def _draw_candidate_boxes(frame, candidates: list[PersonCandidate]):
    boxed = frame.copy()
    for candidate in candidates:
        x1, y1, x2, y2 = candidate.bbox_xyxy
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (35, 189, 120), 2)
        label = f"P{candidate.person_id + 1}"
        cv2.putText(
            boxed,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (35, 189, 120),
            2,
            cv2.LINE_AA,
        )
    return boxed


def _draw_single_box(frame, box_xyxy, label: str):
    boxed = frame.copy()
    x1, y1, x2, y2 = box_xyxy
    cv2.rectangle(boxed, (x1, y1), (x2, y2), (20, 120, 230), 2)
    cv2.putText(
        boxed,
        label,
        (x1, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (20, 120, 230),
        2,
        cv2.LINE_AA,
    )
    return boxed


def _render_compact_image(frame_bgr, caption: str, max_height: int = 420):
    ok, encoded = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), caption=caption, width=520)
        return
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    st.markdown(
        (
            "<figure style='margin:0 0 0.75rem 0;'>"
            "<img style='max-height:{max_height}px;width:auto;max-width:100%;display:block;border-radius:6px;' "
            "src='data:image/jpeg;base64,{payload}' />"
            "<figcaption style='font-size:0.9rem;color:#6b7280;margin-top:0.35rem;'>{caption}</figcaption>"
            "</figure>"
        ).format(max_height=max_height, payload=payload, caption=caption),
        unsafe_allow_html=True,
    )


def _render_compact_video_file(video_path: Path, max_height: int = 920):
    payload = base64.b64encode(video_path.read_bytes()).decode("ascii")
    st.markdown(
        (
            "<div style='display:flex;justify-content:center;'>"
            "<video controls style='max-height:{max_height}px;width:auto;max-width:100%;display:block;border-radius:8px;' "
            "src='data:video/mp4;base64,{payload}'></video>"
            "</div>"
        ).format(max_height=max_height, payload=payload),
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="EdgeSense", layout="wide")
st.title("EdgeSense")
st.write("Upload a skiing video to analyze turn structure and coaching cues.")

uploaded = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv"])
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded.name}") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    first_frame = _read_first_frame(tmp_path)
    candidates = detect_person_candidates_first_frame(tmp_path)
    detector_backend = active_person_detector_backend()
    if detector_backend == "yolo":
        st.caption("Person detector: YOLO")
    elif detector_backend == "hog":
        st.warning("YOLO unavailable, using HOG fallback detector. Detection quality may be lower.")
    selected_candidate = None
    refined_box = None
    selected_id = -1
    if candidates:
        if first_frame is not None:
            st.subheader("Target Skier Selection")
            framed = _draw_candidate_boxes(first_frame, candidates)
            _render_compact_image(framed, caption="First-frame person candidates", max_height=420)

        st.write("Pick the target skier for first-frame person selection.")
        candidate_ids = [c.person_id for c in candidates]
        select_options = (candidate_ids + [-2]) if len(candidate_ids) == 1 else ([-1, -2] + candidate_ids)
        selected_id = st.selectbox(
            "Target skier",
            options=select_options,
            index=0,
            format_func=lambda person_id: (
                "Select a skier"
                if person_id == -1
                else ("Manual target box" if person_id == -2 else f"Person {person_id + 1}")
            ),
        )
        if selected_id >= 0:
            selected_candidate = next((c for c in candidates if c.person_id == selected_id), None)

        frame_h, frame_w = (first_frame.shape[:2] if first_frame is not None else (1080, 1920))
        if selected_id >= -2 and selected_id != -1:
            if selected_candidate is not None:
                default_box = selected_candidate.bbox_xyxy
            else:
                default_box = (
                    int(frame_w * 0.3),
                    int(frame_h * 0.15),
                    int(frame_w * 0.7),
                    int(frame_h * 0.95),
                )

            st.write("Refine target box before analysis.")
            x1, x2 = st.slider(
                "Target X range",
                min_value=0,
                max_value=frame_w,
                value=(default_box[0], default_box[2]),
                step=1,
            )
            y1, y2 = st.slider(
                "Target Y range",
                min_value=0,
                max_value=frame_h,
                value=(default_box[1], default_box[3]),
                step=1,
            )
            refined_box = (x1, y1, max(x1 + 1, x2), max(y1 + 1, y2))

            if first_frame is not None:
                preview = _draw_single_box(first_frame, refined_box, "Target")
                _render_compact_image(preview, caption="Selected target box", max_height=420)

        if first_frame is not None:
            columns = st.columns(min(3, len(candidates)))
            for idx, candidate in enumerate(candidates):
                x1, y1, x2, y2 = candidate.bbox_xyxy
                crop = first_frame[y1:y2, x1:x2]
                with columns[idx % len(columns)]:
                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption=f"Person {candidate.person_id + 1}")
    else:
        st.info("No first-frame person candidates detected. Analysis will run with default single-person flow.")
        selected_id = -2
        selected_candidate = None
        refined_box = None
        if first_frame is not None:
            frame_h, frame_w = first_frame.shape[:2]
            st.write("Set manual target box.")
            x1, x2 = st.slider(
                "Target X range",
                min_value=0,
                max_value=frame_w,
                value=(int(frame_w * 0.3), int(frame_w * 0.7)),
                step=1,
            )
            y1, y2 = st.slider(
                "Target Y range",
                min_value=0,
                max_value=frame_h,
                value=(int(frame_h * 0.15), int(frame_h * 0.95)),
                step=1,
            )
            refined_box = (x1, y1, max(x1 + 1, x2), max(y1 + 1, y2))
            preview = _draw_single_box(first_frame, refined_box, "Target")
            _render_compact_image(preview, caption="Manual target box", max_height=420)

    with st.expander("Preview uploaded video", expanded=False):
        st.video(tmp_path)

    analyze_disabled = len(candidates) > 1 and selected_id == -1
    if st.button("Analyze", disabled=analyze_disabled):
        progress = st.progress(0)
        status = st.empty()

        def on_progress(value: int, message: str):
            progress.progress(min(100, max(0, value)))
            status.write(message)

        selection_payload = None
        if selected_candidate is not None:
            selection_payload = {
                "person_id": selected_candidate.person_id,
                "bbox_xyxy": list(selected_candidate.bbox_xyxy),
                "confidence": selected_candidate.confidence,
                "center_xy": list(selected_candidate.center_xy),
            }
        if selected_id == -2 and refined_box is not None:
            x1, y1, x2, y2 = refined_box
            selection_payload = {
                "person_id": None,
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": 1.0,
                "center_xy": [float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)],
                "manual": True,
            }
        elif selected_candidate is not None and refined_box is not None:
            x1, y1, x2, y2 = refined_box
            selection_payload = {
                "person_id": selected_candidate.person_id,
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": selected_candidate.confidence,
                "center_xy": [float((x1 + x2) * 0.5), float((y1 + y2) * 0.5)],
                "manual": False,
                "detected_bbox_xyxy": list(selected_candidate.bbox_xyxy),
            }

        result = analyze_video(
            tmp_path,
            str(Path("data") / "outputs"),
            progress_cb=on_progress,
            target_selection=selection_payload,
        )
        st.success("Done")
        report_path = Path(result["report_md"])
        preview_path = Path(result.get("annotated_preview", "")) if result.get("annotated_preview") else None
        preview_error = result.get("annotated_preview_error", "")
        if report_path.exists():
            st.subheader("Report")
            st.markdown(report_path.read_text(encoding="utf-8"))
        st.subheader("Annotated Video")
        if preview_path is not None and preview_path.exists():
            left_pad, col_video, right_pad = st.columns([1, 1.6, 1])
            with col_video:
                _render_compact_video_file(preview_path, max_height=920)
        else:
            message = preview_error or "Browser preview could not be generated."
            st.error(f"No video shown: {message}")
