"""
Grader for task_1_easy — Single Applicant Underwriting.

Scoring breakdown (total strictly in (0, 1)):
  0.50 — decision correctness: exact match approve/reject/escalate
  0.25 — interest rate accuracy: within ±0.5% of ground truth (approve only)
  0.20 — risk flag recall: fraction of ground-truth flags identified
  0.05 — risk flag precision: penalty for false-positive flags
"""
from typing import Any


def grade(decisions: list[dict[str, Any]], ground_truths: list[dict]) -> tuple[float, dict]:
    if not decisions or not ground_truths:
        return 0.001, {"error": "empty decisions or ground_truths"}

    # ID-matched lookup
    gt_map: dict[str, dict] = {
        gt["applicant_id"]: gt for gt in ground_truths if gt.get("applicant_id")
    }

    d = decisions[0]
    app_id = d.get("applicant_id", "")
    gt = gt_map.get(app_id) or ground_truths[0]  # fallback to first if ID mismatch

    agent_dec = str(d.get("decision", "")).lower().strip()
    expected = gt["decision"]

    # ── 1. Decision correctness (0.50) ────────────────────────────────────────
    decision_score = 0.50 if agent_dec == expected else 0.0

    # ── 2. Interest rate accuracy (0.25) ──────────────────────────────────────
    rate_score = 0.0
    if expected == "approve" and agent_dec == "approve":
        gt_rate = gt.get("interest_rate")
        agent_rate = d.get("interest_rate")
        if gt_rate is not None and agent_rate is not None:
            try:
                diff = abs(float(agent_rate) - float(gt_rate))
                tolerance = float(gt_rate) * 0.005
                if diff <= tolerance:
                    rate_score = 0.25
                else:
                    rate_score = max(0.0, 0.25 * (1.0 - diff / max(float(gt_rate) * 0.01, 0.001)))
            except (TypeError, ValueError):
                rate_score = 0.0
    elif expected != "approve":
        # Not an approval — rate component not applicable, award full credit
        rate_score = 0.25

    # ── 3. Risk flag recall (0.20) ────────────────────────────────────────────
    gt_flags = set(gt.get("risk_flags", []))
    agent_flags = set(d.get("risk_flags", []))

    if gt_flags:
        recall = len(gt_flags & agent_flags) / len(gt_flags)
        flag_recall_score = 0.20 * recall
    else:
        flag_recall_score = 0.20 if not agent_flags else 0.16

    # ── 4. Risk flag precision (0.05) ─────────────────────────────────────────
    if agent_flags:
        precision = len(gt_flags & agent_flags) / len(agent_flags)
        flag_precision_score = 0.05 * precision
    else:
        flag_precision_score = 0.05 if not gt_flags else 0.0

    raw_total = decision_score + rate_score + flag_recall_score + flag_precision_score

    # ── Clamp strictly to open interval (0, 1) ────────────────────────────────
    total = round(max(0.001, min(0.999, raw_total)), 4)

    info = {
        "n_applicants": 1,
        "applicant_id": app_id,
        "agent_decision": agent_dec,
        "expected_decision": expected,
        "decision_correct": agent_dec == expected,
        "decision_score": round(decision_score, 4),
        "gt_interest_rate": gt.get("interest_rate"),
        "agent_interest_rate": d.get("interest_rate"),
        "rate_score": round(rate_score, 4),
        "gt_flags": sorted(gt_flags),
        "agent_flags": sorted(agent_flags),
        "flag_recall": round(len(gt_flags & agent_flags) / max(len(gt_flags), 1), 4),
        "flag_recall_score": round(flag_recall_score, 4),
        "flag_precision_score": round(flag_precision_score, 4),
        "total_score": total,
    }
    return total, info