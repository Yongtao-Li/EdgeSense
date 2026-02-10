import tempfile
from pathlib import Path

import streamlit as st

from skicoach.pipeline import analyze_video


st.set_page_config(page_title="EdgeSense", layout="wide")
st.title("EdgeSense")
st.write("Upload a skiing video to analyze turn structure and coaching cues.")

uploaded = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv"])
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded.name}") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.expander("Preview uploaded video", expanded=False):
        st.video(tmp_path)
    if st.button("Analyze"):
        progress = st.progress(0)
        status = st.empty()

        def on_progress(value: int, message: str):
            progress.progress(min(100, max(0, value)))
            status.write(message)

        result = analyze_video(tmp_path, str(Path("data") / "outputs"), progress_cb=on_progress)
        st.success("Done")
        report_path = Path(result["report_md"])
        annotated_path = Path(result["annotated_video"])
        if report_path.exists():
            st.subheader("Report")
            st.markdown(report_path.read_text(encoding="utf-8"))
        if annotated_path.exists():
            st.subheader("Annotated Video")
            col_video, _ = st.columns([2, 1])
            video_bytes = annotated_path.read_bytes()
            with col_video:
                st.video(video_bytes, format="video/mp4")
