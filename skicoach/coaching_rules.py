from dataclasses import dataclass
from typing import List

from .config import (
    ANGULATION_DEG_THRESHOLD,
    COM_CENTERED_RATIO_THRESHOLD,
    LATE_COM_RATIO_THRESHOLD,
    LEG_TILT_MIN_DEG,
    OUTSIDE_SKI_OFFSET_THRESHOLD,
)
from .metrics import TurnMetrics


@dataclass
class CoachingIssue:
    code: str
    title: str
    diagnosis: str
    cue: str
    drill: str
    evidence: str
    severity: float
    turn_index: int


def evaluate_issues(metrics: List[TurnMetrics]) -> List[CoachingIssue]:
    issues: List[CoachingIssue] = []
    if not metrics:
        return issues

    for metric in metrics:
        if metric.outside_ski_offset > OUTSIDE_SKI_OFFSET_THRESHOLD or metric.centered_ratio > 0.6:
            sev = max(
                metric.outside_ski_offset - OUTSIDE_SKI_OFFSET_THRESHOLD,
                metric.centered_ratio - COM_CENTERED_RATIO_THRESHOLD,
            )
            issues.append(
                CoachingIssue(
                    code="OUTSIDE_SKI_WEAK",
                    title="Outside ski not dominant",
                    diagnosis="COM remains too centered or away from the outside ski in mid-turn.",
                    cue="Stand on the outside ski.",
                    drill="Inside ski lightening: feather the inside ski tail through arc.",
                    evidence=f"offset={metric.outside_ski_offset:.2f}, centered_ratio={metric.centered_ratio:.2f}",
                    severity=float(max(0.0, min(1.0, sev))),
                    turn_index=metric.turn_index,
                )
            )

        if metric.angulation_deg < ANGULATION_DEG_THRESHOLD and metric.leg_tilt_deg > LEG_TILT_MIN_DEG:
            sev = (ANGULATION_DEG_THRESHOLD - metric.angulation_deg) / max(ANGULATION_DEG_THRESHOLD, 1e-6)
            issues.append(
                CoachingIssue(
                    code="LOW_ANGULATION",
                    title="Low hip angulation / whole-body lean",
                    diagnosis="Legs and torso tip together with limited hip-knee separation near apex.",
                    cue="Knees tip, hips shape.",
                    drill="Garlands focusing on hips moving inside while chest stays quiet.",
                    evidence=f"angulation={metric.angulation_deg:.1f} deg, leg_tilt={metric.leg_tilt_deg:.1f} deg",
                    severity=float(max(0.0, min(1.0, sev))),
                    turn_index=metric.turn_index,
                )
            )

        if metric.com_move_ratio > LATE_COM_RATIO_THRESHOLD:
            sev = (metric.com_move_ratio - LATE_COM_RATIO_THRESHOLD) / max(1.0 - LATE_COM_RATIO_THRESHOLD, 1e-6)
            issues.append(
                CoachingIssue(
                    code="LATE_COM",
                    title="Late COM move at initiation",
                    diagnosis="COM shifts inside too late relative to apex timing.",
                    cue="Hips early.",
                    drill="Early hip touch at initiation before the skis build edge.",
                    evidence=f"timing_ratio={metric.com_move_ratio:.2f}",
                    severity=float(max(0.0, min(1.0, sev))),
                    turn_index=metric.turn_index,
                )
            )

    issues.sort(key=lambda i: i.severity, reverse=True)
    deduped: List[CoachingIssue] = []
    seen = set()
    for issue in issues:
        if issue.code in seen:
            continue
        deduped.append(issue)
        seen.add(issue.code)
    return deduped[:3]
