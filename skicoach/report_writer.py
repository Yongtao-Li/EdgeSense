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

    md_lines = ["# Ski Coaching Report", "", "## Top Issues"]
    if not issues:
        md_lines.append("No issues detected with current thresholds.")
    for issue in issues[:3]:
        md_lines.extend(
            [
                "",
                f"### {issue.code}",
                f"- Diagnosis: {issue.diagnosis}",
                f"- Cue: {issue.cue}",
                f"- Drill: {issue.drill}",
                f"- Evidence: {issue.evidence}",
                f"- Metric calculation: {_issue_calc(issue.code)}",
            ]
        )

    md_lines.extend(["", "## Turn Summary"])
    if not turns_payload:
        md_lines.append("No turns detected.")
    for t in turns_payload:
        md_lines.extend(
            [
                "",
                f"### Turn {t['id']} ({t['direction']})",
                f"- Frames: start={t['frames']['start']}, apex={t['frames']['apex']}, end={t['frames']['end']}",
                f"- Score: {t['scores']['simple']:.2f}",
            ]
        )

    md_lines.extend(["", "## Left vs Right", f"- Left avg outside-ski score: {left_vs_right['left_avg']}"])
    md_lines.append(f"- Right avg outside-ski score: {left_vs_right['right_avg']}")
    md_lines.extend(["", "## Annotated Video", f"- File: {annotated_video_path.name}"])

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path
