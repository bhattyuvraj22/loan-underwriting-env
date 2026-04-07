"""
Grader for task_1_easy — Single Applicant Underwriting.

Scoring breakdown (total = 1.0):
  0.50 — decision correctness: exact match on approve/reject/escalate
  0.25 — interest rate accuracy: for approved applicants, within ±0.5% of ground truth
  0.20 — risk flag recall: fraction of ground-truth flags identified by agent
  0.05 — risk flag precision: penalty for false-positive flags

All ID lookups use applicant_id matching for robustness.
"""
from typing import Any


def grade(decisions: list[dict[str, Any]], ground_truths: list[dict]) -> tuple[float, dict]:
    if not decisions or not ground_truths:
        return 0.0, {"error": "empty decisions or ground_truths"}

    # Build ground truth map by applicant_id
    gt_map: dict[str, dict] = {
        gt["applicant_id"]: gt for gt in ground_truths if gt.get("applicant_id")
    }

    # Task 1 has exactly 1 applicant — grade the single decision
    d = decisions[0]
    app_id = d.get("applicant_id", "")
    gt = gt_map.get(app_id)

    if gt is None:
        # Fallback: try first ground truth if ID doesn't match
        gt = ground_truths[0]

    agent_dec = str(d.get("decision", "")).lower().strip()
    expected_dec = gt["decision"]
    decision_correct = agent_dec == expected_dec

    # ── Decision score (0.50) ──────────────────────────────────────────────────
    decision_score = 0.50 if decision_correct else 0.0

    # ── Interest rate score (0.25) ─────────────────────────────────────────────
    rate_score = 0.0
    rate_detail = {}
    if expected_dec == "approve" and decision_correct:
        gt_rate = gt.get("interest_rate")
        agent_rate = d.get("interest_rate")
        if gt_rate is not None and agent_rate is not None:
            try:
                diff = abs(float(agent_rate) - float(gt_rate))
                tolerance = float(gt_rate) * 0.005  # ±0.5% tolerance
                if diff <= tolerance:
                    rate_score = 0.25
                else:
                    # Partial credit: linear decay up to 1.0% off
                    rate_score = max(0.0, 0.25 * (1.0 - diff / max(float(gt_rate) * 0.01, 0.001)))
                rate_detail = {
                    "agent_rate": agent_rate,
                    "expected_rate": gt_rate,
                    "diff": round(diff, 4),
                    "tolerance": round(tolerance, 4),
                }
            except (TypeError, ValueError):
                rate_detail = {"error": "invalid rate value"}
    elif expected_dec != "approve":
        # Rate not applicable; award full rate credit if agent correctly set null/None
        agent_rate = d.get("interest_rate")
        rate_score = 0.25 if agent_rate is None else 0.10
        rate_detail = {"note": "rate N/A for non-approve decision", "agent_rate": agent_rate}

    # ── Risk flag recall (0.20) ────────────────────────────────────────────────
    gt_flags = set(gt.get("risk_flags", []))
    agent_flags = set(d.get("risk_flags", []))

    if gt_flags:
        recall = len(gt_flags & agent_flags) / len(gt_flags)
    else:
        recall = 1.0 if not agent_flags else 0.8
    flag_recall_score = 0.20 * recall

    # ── Risk flag precision (0.05) — penalise false positives ─────────────────
    if agent_flags:
        precision = len(gt_flags & agent_flags) / len(agent_flags)
    else:
        precision = 1.0 if not gt_flags else 0.0
    flag_precision_score = 0.05 * precision

    total = round(
        min(decision_score + rate_score + flag_recall_score + flag_precision_score, 1.0), 4
    )

    info = {
        "applicant_id": app_id,
        "agent_decision": agent_dec,
        "expected_decision": expected_dec,
        "decision_correct": decision_correct,
        "decision_score": round(decision_score, 4),
        "rate_score": round(rate_score, 4),
        "rate_detail": rate_detail,
        "gt_flags": sorted(gt_flags),
        "agent_flags": sorted(agent_flags),
        "flag_recall": round(recall, 4),
        "flag_precision": round(precision, 4),
        "flag_recall_score": round(flag_recall_score, 4),
        "flag_precision_score": round(flag_precision_score, 4),
        "total_score": total,
    }
    return total, info
