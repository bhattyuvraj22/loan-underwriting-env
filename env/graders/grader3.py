"""
Grader for task_3_hard — Edge Case Portfolio with Escalation.

Scoring breakdown (total = 1.0):
  0.30 — decision accuracy across all 8 applicants
  0.30 — escalation quality: F1 on the escalate class (id-matched, not positional)
  0.20 — risk flag recall: average across all applicants (id-matched)
  0.10 — safety: penalise false approvals of hard-reject applicants
  0.10 — fraud detection: specifically reward catching forced fraud/thin-file cases

All ID lookups use applicant_id matching (not positional zip) to be robust
to agents submitting decisions in any order.
"""
from typing import Any


def grade(decisions: list[dict[str, Any]], ground_truths: list[dict]) -> tuple[float, dict]:
    if not decisions or not ground_truths:
        return 0.0, {"error": "empty decisions or ground_truths"}

    # ── Build ID-keyed ground truth map ───────────────────────────────────────
    gt_map: dict[str, dict] = {}
    for gt in ground_truths:
        aid = gt.get("applicant_id")
        if aid:
            gt_map[aid] = gt

    # Collect all unique applicant IDs (from both decisions and ground truth)
    all_gt_ids = set(gt_map.keys())

    # ── 1. Decision accuracy (0.30) ────────────────────────────────────────────
    correct_count = 0
    hard_reject_false_approvals = 0
    per_decision: list[dict] = []

    for d in decisions:
        app_id = d.get("applicant_id", "")
        gt = gt_map.get(app_id)
        if gt is None:
            per_decision.append({"applicant_id": app_id, "agent": d.get("decision"), "expected": None, "correct": False})
            continue

        agent_dec = str(d.get("decision", "")).lower().strip()
        expected = gt["decision"]
        is_correct = agent_dec == expected

        # Safety tracking: false approvals of should-reject applicants
        if expected == "reject" and agent_dec == "approve":
            hard_reject_false_approvals += 1

        correct_count += int(is_correct)
        per_decision.append({
            "applicant_id": app_id,
            "agent": agent_dec,
            "expected": expected,
            "correct": is_correct,
        })

    n_scored = len(per_decision)
    decision_score = 0.30 * (correct_count / max(n_scored, 1))

    # ── 2. Escalation quality F1 (0.30) — ID-matched ─────────────────────────
    # FIX: Build sets using gt_map lookup per decision, NOT positional zip
    gt_escalate_ids: set[str] = {aid for aid, gt in gt_map.items() if gt["decision"] == "escalate"}
    agent_escalate_ids: set[str] = {
        d.get("applicant_id", "")
        for d in decisions
        if str(d.get("decision", "")).lower().strip() == "escalate"
    }

    tp = len(gt_escalate_ids & agent_escalate_ids)
    precision = tp / max(len(agent_escalate_ids), 1)
    recall = tp / max(len(gt_escalate_ids), 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    escalation_score = 0.30 * f1

    # ── 3. Risk flag recall (0.20) — ID-matched ───────────────────────────────
    # FIX: Use gt_map lookup per decision, NOT positional zip
    flag_recalls: list[float] = []
    for d in decisions:
        app_id = d.get("applicant_id", "")
        gt = gt_map.get(app_id)
        if gt is None:
            continue
        gt_flags = set(gt.get("risk_flags", []))
        agent_flags = set(d.get("risk_flags", []))
        if gt_flags:
            flag_recalls.append(len(gt_flags & agent_flags) / len(gt_flags))
        else:
            # No flags expected: full credit if agent also raises no flags
            flag_recalls.append(1.0 if not agent_flags else 0.8)

    avg_flag_recall = sum(flag_recalls) / max(len(flag_recalls), 1)
    flag_score = 0.20 * avg_flag_recall

    # ── 4. Safety score (0.10): penalise false approvals of hard-rejects ──────
    safety_score = 0.10 * max(0.0, 1.0 - hard_reject_false_approvals / max(n_scored, 1))

    # ── 5. Fraud/thin-file detection bonus (0.10) ─────────────────────────────
    # Specifically reward agents that correctly escalate forced-edge-case applicants.
    # These are the applicants where escalation is required due to fraud_flag OR
    # income_verified=False OR prior_default=True (not just borderline DTI).
    critical_escalate_ids: set[str] = {
        aid for aid, gt in gt_map.items()
        if gt["decision"] == "escalate"
        and any(f in gt.get("risk_flags", []) for f in ("fraud_flag", "unverified_income", "prior_default"))
    }
    if critical_escalate_ids:
        detection_recall = len(critical_escalate_ids & agent_escalate_ids) / len(critical_escalate_ids)
        detection_score = round(0.10 * detection_recall, 4)
    else:
        detection_score = 0.10  # no critical cases in this episode

    raw_total = decision_score + escalation_score + flag_score + safety_score + detection_score
    total = round(max(0.001, min(0.999, raw_total)), 4)

    info = {
        "n_applicants": n_scored,
        "decision_accuracy": round(correct_count / max(n_scored, 1), 4),
        "decision_score": round(decision_score, 4),
        "gt_escalate_count": len(gt_escalate_ids),
        "agent_escalate_count": len(agent_escalate_ids),
        "escalation_tp": tp,
        "escalation_precision": round(precision, 4),
        "escalation_recall": round(recall, 4),
        "escalation_f1": round(f1, 4),
        "escalation_score": round(escalation_score, 4),
        "avg_flag_recall": round(avg_flag_recall, 4),
        "flag_score": round(flag_score, 4),
        "hard_reject_false_approvals": hard_reject_false_approvals,
        "safety_score": round(safety_score, 4),
        "critical_escalate_count": len(critical_escalate_ids),
        "detection_score": round(detection_score, 4),
        "per_decision": per_decision,
    }
    return total, info
