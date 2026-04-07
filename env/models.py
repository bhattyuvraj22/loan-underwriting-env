"""
Typed Pydantic models for Loan Underwriting OpenEnv.
Implements the full OpenEnv interface spec:
  Observation, Action (AgentAction), Reward, StepResult, ResetRequest, TaskInfo
"""
from pydantic import BaseModel, Field
from typing import Any, Optional, Literal


class ApplicantProfile(BaseModel):
    """A single mortgage loan applicant's financial profile."""
    applicant_id: str
    annual_income: float
    monthly_debt: float
    credit_score: int
    loan_amount: float
    property_value: float
    employment_years: float
    employment_type: Literal["salaried", "self_employed", "contract"]
    prior_default: bool
    fraud_flag: bool
    income_verified: bool


class Observation(BaseModel):
    """
    Observation returned by reset() and state().
    Contains all applicant data the agent needs to make decisions.
    """
    task_id: str = Field(description="Which task is active")
    step: int = Field(description="Current step number (0-indexed)")
    max_steps: int = Field(description="Maximum steps allowed before episode force-ends")
    context: dict[str, Any] = Field(description="Task context: applicants list, policy, constraints")
    done: bool = Field(description="True when episode has ended")
    message: str = Field(default="", description="Feedback from last step; empty on reset")


class Reward(BaseModel):
    """
    Typed reward model per OpenEnv spec.
    Provides partial progress signal, not just binary end-of-episode.
    """
    value: float = Field(ge=0.0, le=1.0, description="Scalar reward in [0.0, 1.0]")
    partial: bool = Field(default=False, description="True if reward is for a non-terminal step")
    breakdown: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-component scoring: decision_score, rate_score, flag_score, etc."
    )


class DecisionItem(BaseModel):
    """A single applicant decision within an AgentAction."""
    applicant_id: str
    decision: Literal["approve", "reject", "escalate"]
    interest_rate: Optional[float] = Field(
        default=None,
        description="Required when decision=approve; null otherwise"
    )
    risk_flags: list[str] = Field(
        default_factory=list,
        description="Applicable risk flags: high_dti, low_credit_score, high_ltv, etc."
    )
    reasoning: str = Field(default="", description="Agent's step-by-step justification")


class AgentAction(BaseModel):
    """
    Action submitted to step().
    The agent submits one DecisionItem per applicant in context.applicants.
    """
    task_id: str
    decisions: list[dict[str, Any]]


class StepResult(BaseModel):
    """Full result returned by step(), matching OpenEnv spec."""
    observation: Observation
    reward: float = Field(ge=0.0, le=1.0, description="Scalar reward for this step")
    done: bool
    info: dict[str, Any] = Field(description="Detailed grader breakdown and per-applicant results")


class ResetRequest(BaseModel):
    task_id: str
    seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")


class TaskInfo(BaseModel):
    task_id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    max_steps: int
    reward_range: tuple[float, float] = (0.0, 1.0)
