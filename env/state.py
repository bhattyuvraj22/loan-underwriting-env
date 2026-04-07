"""
Session manager for Loan Underwriting OpenEnv.
Manages episode state, multi-step trajectories, and grader dispatch.
"""
import random

from .graders import grader1, grader2, grader3
from .models import Observation, StepResult, AgentAction, TaskInfo
from .underwriting import make_applicant, compute_ground_truth

TASKS = [
    TaskInfo(
        task_id="task_1_easy",
        name="Single Applicant Underwriting",
        difficulty="easy",
        description=(
            "Evaluate one applicant: compute DTI and LTV, then decide "
            "approve/reject/escalate, assign an interest rate if approved, "
            "and identify all applicable risk flags."
        ),
        max_steps=1,
    ),
    TaskInfo(
        task_id="task_2_medium",
        name="Batch Underwriting with Capital Constraints",
        difficulty="medium",
        description=(
            "Evaluate 6 applicants while respecting a capital budget (total approved "
            "loan value) and a risk cap (max 2 high-DTI approvals). Maximise correct "
            "decisions within constraints."
        ),
        max_steps=1,
    ),
    TaskInfo(
        task_id="task_3_hard",
        name="Edge Case Portfolio with Escalation",
        difficulty="hard",
        description=(
            "Evaluate 8 applicants including forced fraud flags, unverified-income "
            "(thin-file) cases, and borderline DTI applicants. Correctly classify "
            "escalation cases and identify all risk flags."
        ),
        max_steps=1,
    ),
]

_MAX_STEPS = {t.task_id: t.max_steps for t in TASKS}


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, dict] = {}

    def list_tasks(self) -> list[dict]:
        return [t.model_dump() for t in TASKS]

    def reset(self, task_id: str, seed: int = 42) -> Observation:
        """
        Start a new episode. Returns the initial Observation with all applicant data.
        Calling reset() on an active session cleanly replaces it.
        """
        rng = random.Random(seed)

        if task_id == "task_1_easy":
            applicants = [make_applicant(rng, index=0)]
            context: dict = {
                "applicants": [a.model_dump() for a in applicants],
                "policy": (
                    "Evaluate the applicant. "
                    "ESCALATE if: fraud_flag OR income_verified=false OR prior_default OR 0.40<=DTI<=0.45. "
                    "REJECT if: DTI>0.45 OR credit_score<620 OR LTV>0.97. "
                    "APPROVE otherwise. "
                    "Assign interest_rate = 6.5 + max(0,(DTI-0.28)*4) + max(0,(720-credit_score)*0.01) for approvals."
                ),
            }

        elif task_id == "task_2_medium":
            applicants = [make_applicant(rng, index=i) for i in range(6)]
            total_loan = sum(a.loan_amount for a in applicants)
            context = {
                "applicants": [a.model_dump() for a in applicants],
                "capital_budget": int(total_loan * 0.6),
                "risk_cap": 2,
                "policy": (
                    "Evaluate 6 applicants. Apply standard underwriting rules. "
                    "Constraint 1: total approved loan_amount must not exceed capital_budget. "
                    "Constraint 2: at most risk_cap=2 approvals may have DTI > 0.36. "
                    "Escalation/rejection rules same as task_1."
                ),
            }

        elif task_id == "task_3_hard":
            # Build 8 applicants: 4 standard + 1 forced-fraud + 1 forced-thin + 1 forced-borderline + 1 standard
            applicants = (
                [make_applicant(rng, index=i) for i in range(4)]
                + [make_applicant(rng, force_fraud=True, index=4)]
                + [make_applicant(rng, force_thin=True, index=5)]
                + [make_applicant(rng, force_borderline=True, index=6)]
                + [make_applicant(rng, index=7)]
            )
            rng.shuffle(applicants)
            context = {
                "applicants": [a.model_dump() for a in applicants],
                "policy": (
                    "Evaluate 8 applicants including edge cases. "
                    "At least one applicant has fraud_flag=true (must escalate). "
                    "At least one has income_verified=false (must escalate). "
                    "At least one has borderline DTI in [0.40, 0.45] (must escalate). "
                    "Identify ALL risk flags for every applicant regardless of decision. "
                    "Escalation rules: fraud_flag OR income_verified=false OR prior_default OR 0.40<=DTI<=0.45."
                ),
            }

        else:
            raise ValueError(f"Unknown task_id: '{task_id}'. Valid: task_1_easy, task_2_medium, task_3_hard")

        # Compute ground truths — add applicant_id for reliable grader matching
        ground_truths = [
            {**compute_ground_truth(a), "applicant_id": a.applicant_id}
            for a in applicants
        ]

        self._sessions[task_id] = {
            "step": 0,
            "done": False,
            "applicants": applicants,
            "ground_truths": ground_truths,
            "context": context,
            "seed": seed,
        }

        return Observation(
            task_id=task_id,
            step=0,
            max_steps=_MAX_STEPS[task_id],
            context=context,
            done=False,
            message="Episode started. Submit decisions for all applicants in context.applicants.",
        )

    def step(self, action: AgentAction) -> StepResult:
        """
        Submit agent decisions. Returns reward, done=True, and detailed grader breakdown.
        The grader scores all decisions atomically in one step.
        """
        sess = self._sessions.get(action.task_id)
        if not sess:
            raise ValueError(
                f"No active session for '{action.task_id}'. "
                "Call POST /reset with this task_id first."
            )
        if sess["done"]:
            raise ValueError(
                f"Episode for '{action.task_id}' is already done. "
                "Call POST /reset to start a new episode."
            )

        sess["step"] += 1
        gts = sess["ground_truths"]
        ctx = sess["context"]
        decisions = action.decisions

        # Dispatch to appropriate grader
        if action.task_id == "task_1_easy":
            reward, info = grader1.grade(decisions, gts)
        elif action.task_id == "task_2_medium":
            reward, info = grader2.grade(decisions, gts, ctx)
        elif action.task_id == "task_3_hard":
            reward, info = grader3.grade(decisions, gts)
        else:
            raise ValueError(f"Unknown task_id: '{action.task_id}'")

        sess["done"] = True
        sess["last_reward"] = reward
        sess["last_info"] = info

        # Build a human-readable feedback message for the observation
        accuracy = info.get("decision_accuracy", info.get("decision", {}).get("score", 0))
        message = (
            f"Episode complete. reward={reward:.4f}. "
            f"See info for per-component breakdown."
        )

        return StepResult(
            observation=Observation(
                task_id=action.task_id,
                step=sess["step"],
                max_steps=_MAX_STEPS[action.task_id],
                context=ctx,
                done=True,
                message=message,
            ),
            reward=reward,
            done=True,
            info=info,
        )

    def get_state(self, task_id: str) -> Observation:
        """Return current episode state without advancing it."""
        sess = self._sessions.get(task_id)
        if not sess:
            raise ValueError(
                f"No session found for '{task_id}'. "
                "Call POST /reset to start an episode."
            )
        msg = "Episode complete." if sess["done"] else "Waiting for agent decisions."
        return Observation(
            task_id=task_id,
            step=sess["step"],
            max_steps=_MAX_STEPS[task_id],
            context=sess["context"],
            done=sess["done"],
            message=msg,
        )
