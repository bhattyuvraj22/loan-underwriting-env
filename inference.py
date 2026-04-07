"""
Baseline inference script for Loan Underwriting OpenEnv.

Environment variables (all required per competition spec):
  API_BASE_URL   — LLM API endpoint        (e.g. https://api.openai.com/v1)
  MODEL_NAME     — model identifier         (e.g. gpt-4o)
  HF_TOKEN       — API key / HF token       (used as bearer credential) [NO DEFAULT]
  ENV_URL        — environment base URL     (default: http://localhost:7860)

Checklist compliance:
  [x] All LLM calls use OpenAI client configured via API_BASE_URL, MODEL_NAME, HF_TOKEN
  [x] API_BASE_URL and MODEL_NAME have defaults; HF_TOKEN has NO default (raises if missing)
  [x] Stdout logs follow [START]/[STEP]/[END] structured format exactly
  [x] inference.py is in the root directory of the project

Usage:
  export HF_TOKEN=sk-...
  export API_BASE_URL=https://api.openai.com/v1
  export MODEL_NAME=gpt-4o
  python inference.py
  python inference.py --seed 123
"""

import os
import json
import time
import requests
import argparse
from typing import List, Optional
from openai import OpenAI

# ── CLI arguments ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Loan Underwriting OpenEnv baseline inference")
parser.add_argument("--seed", type=int, default=42, help="Random seed for episode reproducibility")
args, _ = parser.parse_known_args()
RUN_SEED: int = args.seed

# ── Configuration from environment variables ───────────────────────────────────
# Checklist: API_BASE_URL and MODEL_NAME have defaults; HF_TOKEN does NOT
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str   = os.getenv("MODEL_NAME", "gpt-4o")
ENV_URL: str      = os.getenv("ENV_URL", "http://localhost:7860")

# Checklist: HF_TOKEN must NOT have a default — raise immediately if missing
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
if not HF_TOKEN:
    raise EnvironmentError(
        "\n[ERROR] HF_TOKEN is not set. This variable is required and has no default.\n"
        "  export HF_TOKEN=your_api_key_or_hf_token\n"
        "Note: Do NOT set a default value for HF_TOKEN per competition checklist."
    )

# ── OpenAI-compatible client — uses HF_TOKEN as the API key ───────────────────
# Checklist: All LLM calls use OpenAI client configured via these variables
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Competition scoring constants ──────────────────────────────────────────────
SUCCESS_SCORE_THRESHOLD = 0.1
BENCHMARK = "loan_underwriting"

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior loan underwriting officer. For every applicant, follow these FIVE PHASES in strict order.

PHASE 1 — CALCULATE (always do this first)
  DTI = (monthly_debt * 12) / annual_income        [keep 4 decimal places]
  LTV = loan_amount / property_value                [keep 4 decimal places]

PHASE 2 — ESCALATION CHECK (highest priority; if any trigger fires, stop here)
  Escalate if ANY of:
    fraud_flag = true
    income_verified = false
    prior_default = true
    0.40 <= DTI <= 0.45  (inclusive both ends)

PHASE 3 — HARD REJECT (only if NOT escalating)
  Reject if ANY of:
    DTI > 0.45
    credit_score < 620
    LTV > 0.97

PHASE 4 — APPROVE (only if not escalated and not rejected)
  interest_rate = round(6.5 + max(0, (DTI-0.28)*4) + max(0, (720-credit_score)*0.01), 2)
  Set interest_rate = null for escalate or reject decisions.

PHASE 5 — RISK FLAGS (check ALL eight for every applicant, regardless of decision)
  "high_dti"             if DTI > 0.36
  "low_credit_score"     if credit_score < 680
  "high_ltv"             if LTV > 0.80
  "short_employment"     if employment_years < 2.0
  "self_employed_income" if employment_type == "self_employed"
  "prior_default"        if prior_default == true
  "fraud_flag"           if fraud_flag == true
  "unverified_income"    if income_verified == false

CRITICAL RULES:
  1. Escalate takes priority over reject and approve
  2. Copy applicant_id EXACTLY from input
  3. interest_rate must be JSON null when decision is not approve
  4. risk_flags is NEVER empty — almost every applicant has at least one flag
  5. Flags apply even to escalated and rejected applicants

OUTPUT — return ONLY valid JSON, no markdown:
{
  "applicant_id": "APP-XXXX-NN",
  "decision": "approve",
  "interest_rate": 7.25,
  "risk_flags": ["high_ltv", "low_credit_score"],
  "reasoning": "DTI=0.3100 LTV=0.8500 credit=650 => no escalation => no hard reject => approve."
}"""


# ── STDOUT logging — competition harness protocol ──────────────────────────────
# Checklist: Stdout logs follow the required structured format (START/STEP/END) exactly

def log_start(task: str, env: str, model: str) -> None:
    """Emit the [START] line — must be first line per episode."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emit one [STEP] line — call immediately after each env.step() / /step POST."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit the [END] line — must always be emitted, even on exception."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Evaluate a single applicant via LLM ───────────────────────────────────────

def evaluate_single_applicant(applicant: dict, policy_context: dict) -> dict:
    """
    Call the LLM to evaluate one applicant.
    Uses OpenAI client (client) with MODEL_NAME and HF_TOKEN — per competition checklist.
    """
    dti = round((applicant.get("monthly_debt", 0) * 12) / max(applicant.get("annual_income", 1), 1), 4)
    ltv = round(applicant.get("loan_amount", 0) / max(applicant.get("property_value", 1), 1), 4)

    constraint_block = ""
    if policy_context.get("capital_budget"):
        constraint_block = f"""
Portfolio constraints (Task 2 — must respect):
  capital_budget = {policy_context['capital_budget']}  (total approved loan_amount must not exceed this)
  risk_cap       = {policy_context.get('risk_cap', 2)}  (max {policy_context.get('risk_cap', 2)} approved applicants may have DTI > 0.36)
"""

    user_prompt = f"""Applicant to evaluate:
{json.dumps(applicant, indent=2)}

Pre-computed ratios (verified correct):
  DTI = {dti}
  LTV = {ltv}

Underwriting rules:
  ESCALATE if: fraud_flag=true OR income_verified=false OR prior_default=true OR 0.40<=DTI<=0.45
  REJECT   if: DTI>0.45 OR credit_score<620 OR LTV>0.97
  APPROVE  otherwise
  Rate = round(6.5 + max(0,(DTI-0.28)*4) + max(0,(720-credit_score)*0.01), 2)

Risk flags to check (include ALL that apply):
  high_dti(DTI>0.36), low_credit_score(<680), high_ltv(LTV>0.80),
  short_employment(<2yr), self_employed_income(employment_type=self_employed),
  prior_default, fraud_flag, unverified_income(income_verified=false)
{constraint_block}
Return a single JSON decision object:
{{
  "applicant_id": "{applicant.get('applicant_id')}",
  "decision": "approve|reject|escalate",
  "interest_rate": <float or null>,
  "risk_flags": [...],
  "reasoning": "show your DTI/LTV calculation and decision logic"
}}"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        seed=RUN_SEED,
    )

    raw = response.choices[0].message.content or "{}"
    result = json.loads(raw)

    if "decisions" in result and isinstance(result["decisions"], list) and result["decisions"]:
        result = result["decisions"][0]

    return result


# ── Run a full task episode ────────────────────────────────────────────────────

def run_task(task_id: str, rewards_list: List[float]) -> tuple:
    """
    Execute one full episode:
      1. POST /reset  -> get applicants + policy context
      2. Evaluate each applicant via LLM (OpenAI client)
      3. POST /step   -> submit decisions, get reward
      4. Append reward to rewards_list (for [END] logging)
    """
    reset_resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id, "seed": RUN_SEED},
        timeout=30,
    )
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    applicants = obs["context"].get("applicants", [])
    policy_context = {k: v for k, v in obs["context"].items() if k != "applicants"}

    all_decisions: list = []
    for applicant in applicants:
        try:
            decision = evaluate_single_applicant(applicant, policy_context)
            decision["applicant_id"] = applicant["applicant_id"]
            all_decisions.append(decision)
        except Exception as exc:
            print(f"  [WARN] LLM failed for {applicant.get('applicant_id')}: {exc}", flush=True)
            all_decisions.append({
                "applicant_id": applicant["applicant_id"],
                "decision": "escalate",
                "interest_rate": None,
                "risk_flags": [],
                "reasoning": f"fallback due to error: {exc}",
            })

    step_resp = requests.post(
        f"{ENV_URL}/step",
        json={"task_id": task_id, "decisions": all_decisions},
        timeout=60,
    )
    step_resp.raise_for_status()
    result = step_resp.json()

    reward = result.get("reward", 0.0)
    done = result.get("done", True)
    error = result.get("error", None)

    rewards_list.append(reward)

    action_summary = f"submitted_{len(all_decisions)}_decisions"
    log_step(step=1, action=action_summary, reward=reward, done=done, error=error)

    return reward, result.get("info", {})


# ── Main entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tasks = ["task_1_easy", "task_2_medium", "task_3_hard"]

    print("=" * 65)
    print("  Loan Underwriting OpenEnv — Baseline Inference")
    print(f"  Model      : {MODEL_NAME}")
    print(f"  API        : {API_BASE_URL}")
    print(f"  Environment: {ENV_URL}")
    print(f"  Seed       : {RUN_SEED}")
    print("=" * 65)

    all_scores: dict = {}
    t_start = time.time()

    for task_id in tasks:
        t0 = time.time()
        rewards_list: List[float] = []
        score = 0.0
        success = False
        steps_taken = 0

        # [START] must be the first structured log line per episode
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        try:
            reward, info = run_task(task_id, rewards_list)

            score = min(max(reward, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD
            steps_taken = 1

            all_scores[task_id] = score
            elapsed = time.time() - t0
            print(f"\n  [{task_id}]  score={score:.4f}  ({elapsed:.1f}s)", flush=True)
            print(json.dumps(info, indent=2), flush=True)

        except requests.HTTPError as exc:
            error_msg = f"HTTP_{exc.response.status_code}"
            print(f"\n  [{task_id}] HTTP {exc.response.status_code}: {exc.response.text[:300]}", flush=True)
            if not rewards_list:
                rewards_list.append(0.0)
                log_step(step=1, action="http_error", reward=0.0, done=True, error=error_msg)
            all_scores[task_id] = 0.0

        except Exception as exc:
            print(f"\n  [{task_id}] ERROR: {exc}", flush=True)
            if not rewards_list:
                rewards_list.append(0.0)
                log_step(step=1, action="exception", reward=0.0, done=True, error=str(exc)[:120])
            all_scores[task_id] = 0.0

        finally:
            # [END] is ALWAYS emitted, even if an exception occurred above
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards_list)

    total_elapsed = time.time() - t_start
    mean_score = sum(all_scores.values()) / max(len(all_scores), 1)

    print(f"\n{'=' * 65}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 65}")
    for tid, s in all_scores.items():
        print(f"  {tid:<22} : {s:.4f}")
    print(f"  {'Mean score':<22} : {mean_score:.4f}")
    print(f"  {'Total time':<22} : {total_elapsed:.1f}s")
    print("=" * 65)
