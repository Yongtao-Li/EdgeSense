import json
from pathlib import Path
from typing import Dict, List

from .coaching_rules import CoachingIssue
from .metrics import TurnMetrics
from .turn_detection import TurnSegment


def _issue_calc(code: str) -> str:
    if code == "OUTSIDE_SKI_WEAK":
        return "|COM_x - outside_ankle_x| / stance_width"
    if code == "LOW_ANGULATION":
        return "|torso_tilt - outside_leg_tilt| near apex"
    if code == "LATE_COM":
        return "(t_com_inside - t_init) / (t_apex - t_init)"
    return "rule-based metric"


def _turn_payload(turn: TurnSegment, metric: TurnMetrics, issues: List[CoachingIssue]) -> Dict[str, object]:
    turn_issues = [i for i in issues if i.turn_index == metric.turn_index]
    simple_score = (metric.outside_ski_score + metric.angulation_score + metric.timing_score) / 3.0
    return {
        "id": metric.turn_index + 1,
        "direction": metric.direction,
        "frames": {"start": turn.start, "apex": turn.apex, "end": turn.end},
        "scores": {
            "outside_ski": round(metric.outside_ski_score, 3),
            "angulation": round(metric.angulation_score, 3),
            "timing": round(metric.timing_score, 3),
            "simple": round(simple_score, 3),
        },
        "flags": [
            {
                "code": issue.code,
                "severity": round(issue.severity, 3),
            }
            for issue in turn_issues
        ],
    }


def _severity_label(severity: float) -> str:
    if severity >= 0.75:
        return "High"
    if severity >= 0.45:
        return "Medium"
    return "Low"


def _safe_fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _top_priority_lines(issues: List[CoachingIssue]) -> List[str]:
    lines: List[str] = []
    if not issues:
        return ["- No major issues detected with current thresholds."]
    for idx, issue in enumerate(issues[:3], start=1):
        lines.extend(
            [
                f"### {idx}) {issue.code} ({_severity_label(issue.severity)})",
                f"- Diagnosis: {issue.diagnosis}",
                f"- Cue: {issue.cue}",
                f"- Drill: {issue.drill}",
                f"- Evidence: {issue.evidence}",
            ]
        )
    return lines


def _turn_table_lines(turns_payload: List[Dict[str, object]]) -> List[str]:
    lines = [
        "| Turn | Dir | Frames (s-a-e) | Outside | Angulation | Timing | Score | Flags |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    if not turns_payload:
        lines.append("| - | - | - | - | - | - | - | No turns detected |")
        return lines

    for t in turns_payload:
        frames = t.get("frames")
        scores = t.get("scores")
        flags = t.get("flags")
        if not isinstance(frames, dict) or not isinstance(scores, dict) or not isinstance(flags, list):
            continue
        flag_bits = []
        for flag in flags:
            if not isinstance(flag, dict):
                continue
            code = str(flag.get("code", ""))
            severity = float(flag.get("severity", 0.0))
            flag_bits.append(f"{code}({_severity_label(severity)})")
        flag_text = ", ".join(flag_bits) if flag_bits else "None"
        lines.append(
            "| {turn_id} | {direction} | {start}-{apex}-{end} | {outside:.2f} | {ang:.2f} | {timing:.2f} | {simple:.2f} | {flags} |".format(
                turn_id=t.get("id", "?"),
                direction=t.get("direction", "?"),
                start=frames.get("start", "?"),
                apex=frames.get("apex", "?"),
                end=frames.get("end", "?"),
                outside=float(scores.get("outside_ski", 0.0)),
                ang=float(scores.get("angulation", 0.0)),
                timing=float(scores.get("timing", 0.0)),
                simple=float(scores.get("simple", 0.0)),
                flags=flag_text,
            )
        )
    return lines


def _next_session_lines(issues: List[CoachingIssue]) -> List[str]:
    if not issues:
        return ["- Keep current cues and collect one more clip for trend tracking."]

    drills = []
    seen = set()
    for issue in issues:
        if issue.drill in seen:
            continue
        seen.add(issue.drill)
        drills.append((issue.code, issue.drill))
        if len(drills) >= 3:
            break

    lines = []
    for idx, (code, drill) in enumerate(drills, start=1):
        lines.append(f"{idx}. {code}: {drill}")
    return lines


def write_reports(
    output_dir: Path,
    video_meta: Dict[str, object],
    turns: List[TurnSegment],
    metrics: List[TurnMetrics],
    issues: List[CoachingIssue],
    annotated_video_path: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"

    left_scores = [m.outside_ski_score for m in metrics if m.direction == "left"]
    right_scores = [m.outside_ski_score for m in metrics if m.direction == "right"]
    left_vs_right = {
        "left_avg": round(sum(left_scores) / len(left_scores), 3) if left_scores else None,
        "right_avg": round(sum(right_scores) / len(right_scores), 3) if right_scores else None,
    }

    turns_payload = []
    for turn, metric in zip(turns, metrics):
        turns_payload.append(_turn_payload(turn, metric, issues))

    payload = {
        "video": video_meta,
        "turns": turns_payload,
        "left_vs_right": left_vs_right,
        "top_issues": [
            {
                "code": issue.code,
                "severity": round(issue.severity, 3),
                "cue": issue.cue,
                "drill": issue.drill,
                "diagnosis": issue.diagnosis,
                "evidence": issue.evidence,
                "calculation": _issue_calc(issue.code),
            }
            for issue in issues[:3]
        ],
        "annotated_video": annotated_video_path.name,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    tracking = video_meta.get("tracking") if isinstance(video_meta, dict) else None
    tracking_summary = tracking.get("summary") if isinstance(tracking, dict) else None
    lock_ratio = tracking_summary.get("lock_ratio") if isinstance(tracking_summary, dict) else None
    reacquire_count = tracking_summary.get("reacquire_count") if isinstance(tracking_summary, dict) else None
    max_lost = tracking_summary.get("max_lost_streak") if isinstance(tracking_summary, dict) else None
    pose_stats = video_meta.get("pose_stats") if isinstance(video_meta, dict) else None
    pose_drift_frames = pose_stats.get("drift_frames") if isinstance(pose_stats, dict) else None
    pose_fallback_frames = pose_stats.get("fallback_frames") if isinstance(pose_stats, dict) else None

    md_lines = [
        "# Ski Coaching Report",
        "",
        "## Session Snapshot",
        "| Item | Value |",
        "| --- | --- |",
        f"| Video | {Path(str(video_meta.get('path', 'unknown'))).name} |",
        f"| View | {video_meta.get('view', 'unknown')} |",
        f"| FPS | {_safe_fmt(video_meta.get('fps', 'n/a'))} |",
        f"| Frames | {video_meta.get('frame_count', 'n/a')} |",
        f"| Turns Detected | {len(turns_payload)} |",
        f"| Pose Mode | {video_meta.get('pose_mode', 'full_frame')} |",
        f"| Detector | {video_meta.get('person_detector_backend', 'unknown')} |",
        f"| Tracking Lock Ratio | {_safe_fmt(lock_ratio)} |",
        f"| Reacquire Count | {_safe_fmt(reacquire_count, 0)} |",
        f"| Max Lost Streak | {_safe_fmt(max_lost, 0)} |",
        f"| Pose Drift Frames | {_safe_fmt(pose_drift_frames, 0)} |",
        f"| Pose Fallback Frames | {_safe_fmt(pose_fallback_frames, 0)} |",
        "",
        "## Top Priorities",
    ]

    md_lines.extend(_top_priority_lines(issues))

    md_lines.extend([
        "",
        "## Turn Dashboard",
    ])
    md_lines.extend(_turn_table_lines(turns_payload))

    imbalance = "Balanced"
    if left_vs_right["left_avg"] is not None and left_vs_right["right_avg"] is not None:
        delta = float(left_vs_right["left_avg"] - left_vs_right["right_avg"])
        if abs(delta) > 0.08:
            imbalance = "Left stronger" if delta > 0 else "Right stronger"
    md_lines.extend(
        [
            "",
            "## Left vs Right",
            "| Metric | Left | Right | Note |",
            "| --- | ---: | ---: | --- |",
            "| Outside-ski control | {left} | {right} | {note} |".format(
                left=_safe_fmt(left_vs_right["left_avg"]),
                right=_safe_fmt(left_vs_right["right_avg"]),
                note=imbalance,
            ),
            "",
            "## Next Session Plan",
        ]
    )
    md_lines.extend(_next_session_lines(issues))

    md_lines.extend(
        [
            "",
            "## Confidence Notes",
            "- Tracking and pose values are strongest when lock ratio is high and lost streak is low.",
            "- Use this report for trend tracking across multiple clips, not one-off verdicts.",
            "",
            "## Metric Definitions",
        ]
    )
    if issues:
        for issue in issues[:3]:
            md_lines.append(f"- {issue.code}: {_issue_calc(issue.code)}")
    else:
        md_lines.append("- No issue formulas shown because no issues were flagged.")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path
