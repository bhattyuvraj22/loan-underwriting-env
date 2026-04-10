"""
Grader for task_2_medium — Batch Underwriting with Capital Constraints.

Scoring breakdown (total = 1.0):
  0.40 — decision accuracy: per-applicant correct decision rate (ID-matched)
  0.25 — capital budget constraint: total approved loan_amount vs capital_budget
  0.20 — risk cap constraint: at most 2 approvals with DTI > 0.36
  0.10 — interest rate accuracy: for correctly approved applicants
  0.05 — fraud/safety bonus: correctly escalated all mandatory-escalation applicants

All ID lookups use applicant_id matching (not positional zip).
"""
from typing import Any


def grade(
    decisions: list[dict[str, Any]],
    ground_truths: list[dict],
    context: dict,
) -> tuple[float, dict]:
    if not decisions or not ground_truths:
        return 0.0, {"error": "empty decisions or ground_truths"}

    # ── Build ID-keyed maps ────────────────────────────────────────────────────
    gt_map: dict[str, dict] = {
        gt["applicant_id"]: gt for gt in ground_truths if gt.get("applicant_id")
    }

    # Build applicant profile map from context for DTI/loan_amount lookups
    applicant_map: dict[str, dict] = {}
    for app in context.get("applicants", []):
        applicant_map[app["applicant_id"]] = app

    capital_budget: float = float(context.get("capital_budget", float("inf")))
    risk_cap: int = int(context.get("risk_cap", 2))

    # ── 1. Decision accuracy (0.40) ────────────────────────────────────────────
    correct_count = 0
    per_decision: list[dict] = []

    for d in decisions:
        app_id = d.get("applicant_id", "")
        gt = gt_map.get(app_id)
        if gt is None:
            per_decision.append({
                "applicant_id": app_id, "agent": d.get("decision"),
                "expected": None, "correct": False,
            })
            continue
        agent_dec = str(d.get("decision", "")).lower().strip()
        expected = gt["decision"]
        is_correct = agent_dec == expected
        correct_count += int(is_correct)
        per_decision.append({
            "applicant_id": app_id, "agent": agent_dec,
            "expected": expected, "correct": is_correct,
        })

    n_scored = max(len(per_decision), 1)
    decision_score = 0.40 * (correct_count / n_scored)

    # ── 2. Capital budget constraint (0.25) ────────────────────────────────────
    total_approved_loan = 0.0
    for d in decisions:
        app_id = d.get("applicant_id", "")
        agent_dec = str(d.get("decision", "")).lower().strip()
        if agent_dec == "approve":
            app = applicant_map.get(app_id, {})
            total_approved_loan += float(app.get("loan_amount", 0))

    if total_approved_loan <= capital_budget:
        budget_score = 0.25
    else:
        # Proportional penalty for going over budget
        overage_ratio = (total_approved_loan - capital_budget) / max(capital_budget, 1)
        budget_score = max(0.0, 0.25 * (1.0 - min(overage_ratio, 1.0)))

    # ── 3. Risk cap constraint (0.20) ──────────────────────────────────────────
    high_dti_approvals = 0
    for d in decisions:
        app_id = d.get("applicant_id", "")
        agent_dec = str(d.get("decision", "")).lower().strip()
        if agent_dec == "approve":
            gt = gt_map.get(app_id, {})
            dti = gt.get("dti", 0)
            if dti > 0.36:
                high_dti_approvals += 1

    if high_dti_approvals <= risk_cap:
        risk_cap_score = 0.20
    else:
        excess = high_dti_approvals - risk_cap
        risk_cap_score = max(0.0, 0.20 * (1.0 - excess / max(risk_cap, 1)))

    # ── 4. Interest rate accuracy (0.10) ───────────────────────────────────────
    rate_scores: list[float] = []
    for d in decisions:
        app_id = d.get("applicant_id", "")
        gt = gt_map.get(app_id)
        if gt is None:
            continue
        agent_dec = str(d.get("decision", "")).lower().strip()
        expected = gt["decision"]
        if expected == "approve" and agent_dec == "approve":
            gt_rate = gt.get("interest_rate")
            agent_rate = d.get("interest_rate")
            if gt_rate is not None and agent_rate is not None:
                try:
                    diff = abs(float(agent_rate) - float(gt_rate))
                    tolerance = float(gt_rate) * 0.005
                    if diff <= tolerance:
                        rate_scores.append(1.0)
                    else:
                        rate_scores.append(max(0.0, 1.0 - diff / max(float(gt_rate) * 0.01, 0.001)))
                except (TypeError, ValueError):
                    rate_scores.append(0.0)

    avg_rate_score = (sum(rate_scores) / len(rate_scores)) if rate_scores else 1.0
    interest_rate_score = 0.10 * avg_rate_score

    # ── 5. Fraud/safety bonus (0.05) ───────────────────────────────────────────
    # Full bonus if all mandatory-escalation applicants were escalated
    mandatory_escalate_ids = {
        aid for aid, gt in gt_map.items() if gt["decision"] == "escalate"
    }
    agent_escalate_ids = {
        d.get("applicant_id", "")
        for d in decisions
        if str(d.get("decision", "")).lower().strip() == "escalate"
    }
    if mandatory_escalate_ids:
        safety_recall = len(mandatory_escalate_ids & agent_escalate_ids) / len(mandatory_escalate_ids)
        safety_score = 0.05 * safety_recall
    else:
        safety_score = 0.05  # no mandatory escalations in this episode

    raw_total = decision_score + budget_score + risk_cap_score + interest_rate_score + safety_score
    total = round(max(0.001, min(0.999, raw_total)), 4)

    info = {
        "n_applicants": n_scored,
        "decision_accuracy": round(correct_count / n_scored, 4),
        "decision_score": round(decision_score, 4),
        "total_approved_loan": round(total_approved_loan, 2),
        "capital_budget": capital_budget,
        "budget_ok": total_approved_loan <= capital_budget,
        "budget_score": round(budget_score, 4),
        "high_dti_approvals": high_dti_approvals,
        "risk_cap": risk_cap,
        "risk_cap_ok": high_dti_approvals <= risk_cap,
        "risk_cap_score": round(risk_cap_score, 4),
        "avg_rate_accuracy": round(avg_rate_score, 4),
        "interest_rate_score": round(interest_rate_score, 4),
        "mandatory_escalate_count": len(mandatory_escalate_ids),
        "agent_escalate_count": len(agent_escalate_ids & mandatory_escalate_ids),
        "safety_score": round(safety_score, 4),
        "total_score": total,
        "per_decision": per_decision,
    }
    return total, info
