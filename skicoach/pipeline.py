from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import cv2

from .config import VIDEO_CODEC_CANDIDATES
from .coaching_rules import evaluate_issues
from .cropping import compute_crop_boxes, crop_frame
from .metrics import compute_turn_metrics
from .overlay import overlay_frame
from .pose_provider import MediaPipePoseProvider, mask_low_visibility
from .report_writer import write_reports
from .smoothing import smooth_keypoints
from .turn_detection import TurnSegment, detect_turns
from .view_detection import detect_view


def _phase_for_frame(frame_idx: int, turn: TurnSegment) -> str:
    if frame_idx <= turn.start + (turn.apex - turn.start) // 2:
        return "initiation"
    if frame_idx <= turn.apex + (turn.end - turn.apex) // 2:
        return "apex"
    return "finish"


def _turn_lookup(turns: list[TurnSegment], frame_idx: int) -> Tuple[Optional[int], Optional[TurnSegment]]:
    for i, turn in enumerate(turns):
        if turn.start <= frame_idx <= turn.end:
            return i, turn
    return None, None


def _even_size(width: int, height: int) -> Tuple[int, int]:
    even_w = width if width % 2 == 0 else max(2, width - 1)
    even_h = height if height % 2 == 0 else max(2, height - 1)
    return even_w, even_h


def _create_video_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    for codec in VIDEO_CODEC_CANDIDATES:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    tried = ", ".join(VIDEO_CODEC_CANDIDATES)
    raise RuntimeError(f"Unable to open VideoWriter with codecs: {tried}")


def analyze_video(
    video_path: str,
    output_root: str,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, str]:
    def report(step: int, message: str):
        if progress_cb:
            progress_cb(step, message)

    report(5, "Loading video and pose model")
    provider = MediaPipePoseProvider()
    report(15, "Running pose estimation")
    pose_seq = provider.read_video(video_path)
    if not pose_seq.frames:
        raise ValueError("No frames found in video")

    report(30, "Smoothing keypoints")
    masked = mask_low_visibility(pose_seq.keypoints, pose_seq.visibility)
    smoothed = smooth_keypoints(masked)

    report(45, "Detecting turns")
    turns = detect_turns(smoothed)

    report(60, "Computing metrics and coaching rules")
    metrics = compute_turn_metrics(smoothed, turns)
    issues = evaluate_issues(metrics)

    video_name = Path(video_path).stem
    output_dir = Path(output_root) / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    report(72, "Computing skier crop window")
    crop_boxes = compute_crop_boxes(smoothed, pose_seq.visibility, pose_seq.frames[0].shape)

    report(85, "Rendering annotated video")
    first_crop = crop_frame(pose_seq.frames[0], crop_boxes[0])
    ch, cw = first_crop.shape[:2]
    cw, ch = _even_size(cw, ch)
    annotated_video_path = output_dir / "annotated.mp4"
    writer = _create_video_writer(annotated_video_path, pose_seq.fps, cw, ch)

    for i, frame in enumerate(pose_seq.frames):
        turn_id, turn = _turn_lookup(turns, i)
        phase = _phase_for_frame(i, turn) if turn else None
        drawn = overlay_frame(frame, smoothed[i], pose_seq.visibility[i], phase=phase, turn_id=turn_id)
        cropped = crop_frame(drawn, crop_boxes[i])
        if cropped.shape[0] != ch or cropped.shape[1] != cw:
            cropped = cv2.resize(cropped, (cw, ch), interpolation=cv2.INTER_LINEAR)
        writer.write(cropped)
    writer.release()

    report(95, "Writing reports")
    video_meta = {
        "path": video_path,
        "fps": round(float(pose_seq.fps), 2),
        "view": detect_view(smoothed),
        "frame_count": len(pose_seq.frames),
    }
    json_path, md_path = write_reports(
        output_dir=output_dir,
        video_meta=video_meta,
        turns=turns,
        metrics=metrics,
        issues=issues,
        annotated_video_path=annotated_video_path,
    )

    report(100, "Done")
    return {
        "report_json": str(json_path),
        "report_md": str(md_path),
        "annotated_video": str(annotated_video_path),
        "output_dir": str(output_dir),
    }
