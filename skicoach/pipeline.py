import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

from .config import (
    POSE_CROP_MARGIN_X,
    POSE_CROP_MARGIN_Y,
    RENDER_MAX_UPSCALE,
    RENDER_MIN_CROP_HEIGHT,
    RENDER_MIN_CROP_WIDTH,
    RENDER_SIZE_PERCENTILE,
    RENDER_WARMUP_FRAMES,
    RENDER_WARMUP_SCALE,
    VIDEO_CODEC_CANDIDATES,
)
from .coaching_rules import evaluate_issues
from .cropping import compute_crop_boxes, crop_frame
from .metrics import compute_turn_metrics
from .overlay import overlay_frame
from .pose_provider import MediaPipePoseProvider, active_person_detector_backend, mask_low_visibility, read_video_frames
from .report_writer import write_reports
from .smoothing import smooth_keypoints
from .tracking import track_target_boxes
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


def _transcode_web_preview(input_path: Path, output_path: Path) -> Tuple[bool, str]:
    ffmpeg_path = None
    try:
        import imageio_ffmpeg  # type: ignore

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_path = None

    if ffmpeg_path is None:
        ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return False, "No ffmpeg runtime found. Install imageio-ffmpeg or ffmpeg on PATH."

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "ffmpeg transcoding failed").strip()
        return False, err
    if not output_path.exists() or output_path.stat().st_size <= 0:
        return False, "ffmpeg finished but preview file is missing or empty"
    return True, "ok"


def _target_render_size(crop_boxes: list[np.ndarray]) -> Tuple[int, int]:
    widths = [max(2, int(box[2] - box[0])) for box in crop_boxes]
    heights = [max(2, int(box[3] - box[1])) for box in crop_boxes]
    target_w = int(np.percentile(np.asarray(widths, dtype=np.float32), RENDER_SIZE_PERCENTILE))
    target_h = int(np.percentile(np.asarray(heights, dtype=np.float32), RENDER_SIZE_PERCENTILE))
    return _even_size(max(2, target_w), max(2, target_h))


def _expand_box_for_upscale_cap(
    box: np.ndarray,
    frame_shape: Tuple[int, int, int],
    target_w: int,
    target_h: int,
    max_upscale: float,
) -> np.ndarray:
    if max_upscale <= 1.0:
        return box.astype(np.int32)

    height, width = frame_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    min_w = int(np.ceil(float(target_w) / max_upscale))
    min_h = int(np.ceil(float(target_h) / max_upscale))
    need_w = max(box_w, min_w)
    need_h = max(box_h, min_h)

    if need_w <= box_w and need_h <= box_h:
        return box.astype(np.int32)

    need_w = min(width, max(2, need_w))
    need_h = min(height, max(2, need_h))
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    nx1 = int(round(cx - (need_w * 0.5)))
    ny1 = int(round(cy - (need_h * 0.5)))
    nx1 = max(0, min(nx1, width - need_w))
    ny1 = max(0, min(ny1, height - need_h))
    nx2 = nx1 + need_w
    ny2 = ny1 + need_h
    return np.array([nx1, ny1, nx2, ny2], dtype=np.int32)


def _resize_interpolation(src_w: int, src_h: int, dst_w: int, dst_h: int) -> int:
    if dst_w > src_w or dst_h > src_h:
        return cv2.INTER_CUBIC
    return cv2.INTER_AREA


def _expand_box_around_center(
    box: np.ndarray,
    frame_shape: Tuple[int, int, int],
    target_w: int,
    target_h: int,
) -> np.ndarray:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    need_w = min(width, max(box_w, target_w))
    need_h = min(height, max(box_h, target_h))

    if need_w <= box_w and need_h <= box_h:
        return box.astype(np.int32)

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    nx1 = int(round(cx - (need_w * 0.5)))
    ny1 = int(round(cy - (need_h * 0.5)))
    nx1 = max(0, min(nx1, width - need_w))
    ny1 = max(0, min(ny1, height - need_h))
    nx2 = nx1 + need_w
    ny2 = ny1 + need_h
    return np.array([nx1, ny1, nx2, ny2], dtype=np.int32)


def _apply_crop_floor_and_warmup(box: np.ndarray, frame_shape: Tuple[int, int, int], frame_idx: int) -> np.ndarray:
    floor_box = _expand_box_around_center(
        box,
        frame_shape,
        target_w=max(2, int(RENDER_MIN_CROP_WIDTH)),
        target_h=max(2, int(RENDER_MIN_CROP_HEIGHT)),
    )

    warmup_frames = max(0, int(RENDER_WARMUP_FRAMES))
    warmup_scale = max(1.0, float(RENDER_WARMUP_SCALE))
    if warmup_frames <= 0 or warmup_scale <= 1.0:
        return floor_box

    progress = min(1.0, float(frame_idx) / float(warmup_frames))
    frame_scale = 1.0 + ((warmup_scale - 1.0) * (1.0 - progress))
    if frame_scale <= 1.0:
        return floor_box

    scaled_w = int(round((floor_box[2] - floor_box[0]) * frame_scale))
    scaled_h = int(round((floor_box[3] - floor_box[1]) * frame_scale))
    return _expand_box_around_center(floor_box, frame_shape, target_w=scaled_w, target_h=scaled_h)


def _strict_pose_box(box: np.ndarray, frame_shape: Tuple[int, int, int]) -> np.ndarray:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
    bw = max(2, x2 - x1)
    bh = max(2, y2 - y1)
    mx = int(round(bw * float(POSE_CROP_MARGIN_X)))
    my = int(round(bh * float(POSE_CROP_MARGIN_Y)))
    nx1 = max(0, x1 - mx)
    ny1 = max(0, y1 - my)
    nx2 = min(width, x2 + mx)
    ny2 = min(height, y2 + my)
    if nx2 <= nx1:
        nx2 = min(width, nx1 + 2)
    if ny2 <= ny1:
        ny2 = min(height, ny1 + 2)
    return np.array([nx1, ny1, nx2, ny2], dtype=np.int32)


def _clip_target_box(target_selection: Optional[Dict[str, object]], frame_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    if not target_selection:
        return None
    raw_box = target_selection.get("bbox_xyxy")
    if not isinstance(raw_box, list) or len(raw_box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in raw_box]
    except (TypeError, ValueError):
        return None

    height, width = frame_shape[:2]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def analyze_video(
    video_path: str,
    output_root: str,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    target_selection: Optional[Dict[str, object]] = None,
) -> Dict[str, str]:
    def report(step: int, message: str):
        if progress_cb:
            progress_cb(step, message)

    report(5, "Loading video and pose model")
    provider = MediaPipePoseProvider()
    fps, frames = read_video_frames(video_path)
    if not frames:
        raise ValueError("No frames found in video")

    video_name = Path(video_path).stem
    output_dir = Path(output_root) / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_box = _clip_target_box(target_selection, frames[0].shape)
    tracking_payload = None
    tracking_states: list[str] = []
    tracking_confidences: list[float] = []
    crop_boxes: list[np.ndarray] = []
    pose_crop_boxes: Optional[list[np.ndarray]] = None
    pose_mode = "full_frame"

    if selected_box is not None:
        report(12, "Locking target tracking and reacquire")
        tracking_result = track_target_boxes(frames, selected_box)
        crop_boxes = tracking_result["boxes"]
        tracking_states = tracking_result["states"]
        tracking_confidences = tracking_result["confidences"]
        tracking_summary = tracking_result["summary"]
        tracking_payload = {
            "enabled": True,
            "summary": tracking_summary,
            "states": tracking_states,
            "confidences": tracking_confidences,
        }
        pose_crop_boxes = [_strict_pose_box(crop_boxes[i], frames[i].shape) for i in range(len(crop_boxes))]
        pose_mode = "tracked_crop"

    report(20, "Running pose estimation")
    pose_seq = provider.infer_frames(
        frames=frames,
        fps=fps,
        crop_boxes=pose_crop_boxes,
        expected_boxes=crop_boxes if selected_box is not None else None,
    )
    pose_stats = provider.last_infer_stats

    report(30, "Smoothing keypoints")
    masked = mask_low_visibility(pose_seq.keypoints, pose_seq.visibility)
    smoothed = smooth_keypoints(masked)

    report(45, "Detecting turns")
    turns = detect_turns(smoothed)

    report(60, "Computing metrics and coaching rules")
    metrics = compute_turn_metrics(smoothed, turns)
    issues = evaluate_issues(metrics)

    if selected_box is None:
        report(72, "Computing skier crop window")
        crop_boxes = compute_crop_boxes(
            smoothed,
            pose_seq.visibility,
            pose_seq.frames[0].shape,
            initial_box=selected_box,
        )
        tracking_payload = {
            "enabled": False,
            "summary": {
                "lock_ratio": None,
                "reacquire_count": 0,
                "max_lost_streak": 0,
                "mean_confidence": None,
            },
            "states": tracking_states,
            "confidences": tracking_confidences,
        }
    else:
        report(72, "Using tracked skier crop window")

    if not crop_boxes:
        raise ValueError("No crop boxes available for rendering")

    report(85, "Rendering annotated video")
    cw, ch = _target_render_size(crop_boxes)
    annotated_video_path = output_dir / "annotated.mp4"
    writer = _create_video_writer(annotated_video_path, pose_seq.fps, cw, ch)

    for i, frame in enumerate(pose_seq.frames):
        turn_id, turn = _turn_lookup(turns, i)
        phase = _phase_for_frame(i, turn) if turn else None
        frame_tracking_state = tracking_states[i] if i < len(tracking_states) else None
        drawn = overlay_frame(
            frame,
            smoothed[i],
            pose_seq.visibility[i],
            phase=phase,
            turn_id=turn_id,
            target_box=crop_boxes[i] if selected_box is not None else None,
            tracking_state=frame_tracking_state,
        )
        frame_box = _apply_crop_floor_and_warmup(crop_boxes[i], frame.shape, frame_idx=i)
        frame_box = _expand_box_for_upscale_cap(
            frame_box,
            frame.shape,
            target_w=cw,
            target_h=ch,
            max_upscale=RENDER_MAX_UPSCALE,
        )
        cropped = crop_frame(drawn, frame_box)
        if cropped.shape[0] != ch or cropped.shape[1] != cw:
            interp = _resize_interpolation(cropped.shape[1], cropped.shape[0], cw, ch)
            cropped = cv2.resize(cropped, (cw, ch), interpolation=interp)
        writer.write(cropped)
    writer.release()

    report(90, "Creating browser-compatible video preview")
    preview_path = output_dir / "annotated_preview.mp4"
    preview_ok, preview_message = _transcode_web_preview(annotated_video_path, preview_path)
    if preview_ok:
        try:
            annotated_video_path.unlink(missing_ok=True)
        except OSError:
            pass
    report_video_path = preview_path if preview_ok else annotated_video_path

    report(95, "Writing reports")
    video_meta = {
        "path": video_path,
        "fps": round(float(pose_seq.fps), 2),
        "view": detect_view(smoothed),
        "frame_count": len(pose_seq.frames),
        "pose_mode": pose_mode,
        "pose_stats": pose_stats,
        "person_detector_backend": active_person_detector_backend(),
        "tracking": tracking_payload,
    }
    json_path, md_path = write_reports(
        output_dir=output_dir,
        video_meta=video_meta,
        turns=turns,
        metrics=metrics,
        issues=issues,
        annotated_video_path=report_video_path,
    )

    target_metadata_path = output_dir / "target_selection.json"
    target_metadata = {
        "selected": bool(target_selection),
        "target_selection": target_selection,
        "applied_initial_box_xyxy": selected_box.astype(int).tolist() if selected_box is not None else None,
        "person_detector_backend": active_person_detector_backend(),
        "pose_mode": pose_mode,
        "pose_stats": pose_stats,
        "render": {
            "target_size": [cw, ch],
            "max_upscale": RENDER_MAX_UPSCALE,
            "size_percentile": RENDER_SIZE_PERCENTILE,
            "min_crop_floor": [RENDER_MIN_CROP_WIDTH, RENDER_MIN_CROP_HEIGHT],
            "warmup_frames": RENDER_WARMUP_FRAMES,
            "warmup_scale": RENDER_WARMUP_SCALE,
            "pose_crop_margin": [POSE_CROP_MARGIN_X, POSE_CROP_MARGIN_Y],
        },
        "web_preview": {
            "available": preview_ok,
            "path": str(preview_path) if preview_ok else None,
            "error": None if preview_ok else preview_message,
        },
        "tracking": tracking_payload,
    }
    target_metadata_path.write_text(json.dumps(target_metadata, indent=2), encoding="utf-8")

    report(100, "Done")
    return {
        "report_json": str(json_path),
        "report_md": str(md_path),
        "annotated_video": str(annotated_video_path),
        "annotated_preview": str(preview_path) if preview_ok else "",
        "annotated_preview_error": "" if preview_ok else preview_message,
        "output_dir": str(output_dir),
        "target_selection": str(target_metadata_path),
    }
